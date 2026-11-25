#pragma once

#include <Eigen/Dense>
#include <kalmanif/kalmanif.h>
#include <manif/Bundle.h>
#include <manif/Rn.h>
#include <manif/SE_2_3.h>
#include <vector>

#include "radom/types.hh"

namespace radom {

// 15-DoF Composite Manifold: SE_2(3) [p, R, v] x R^3 [b_g] x R^3 [b_a]
// Tangent space ordering: [dp(3), dtheta(3), dv(3), db_g(3), db_a(3)]
using RioState = manif::Bundle<double, manif::SE_2_3, manif::R3, manif::R3>;

// 12-DoF control/noise input: [gyro_raw(3); accel_raw(3); 0(3); 0(3)]
// Trailing 6 channels correspond to gyro/accel bias random-walk noise inputs
// so kalmanif's Jacobian<State, Control> (15x12) and Covariance<Control> (12x12) align.
using ImuControl = Eigen::Matrix<double, 12, 1>;
using RadarMeas = Eigen::Vector3d;  // [v_rx, v_ry, v_rz]

struct RioConfig {
  // Extrinsics: Radar frame expressed in IMU body frame (T_br = T_il * T_rl^{-1})
  Eigen::Matrix3d R_br = (Eigen::Matrix3d() <<
      0.999764256813, -0.017112564047,  0.013363792560,
      0.017209555548,  0.999826147114, -0.007176820029,
     -0.013238655433,  0.007405113073,  0.999884944532).finished();

  Eigen::Vector3d l_br = Eigen::Vector3d(
      2.648919382568, 0.029987617115, -1.109916020193);

  // Gravity in ENU world frame [m/s^2]
  Eigen::Vector3d gravity = Eigen::Vector3d(0.0, 0.0, -9.80665);

  // Continuous-time IMU noise densities
  double gyro_noise_std = 1.5e-3;       // rad / (s * sqrt(Hz))
  double accel_noise_std = 2.0e-2;      // m / (s^2 * sqrt(Hz))
  double gyro_bias_rw_std = 1.0e-4;     // rad / (s^2 * sqrt(Hz))
  double accel_bias_rw_std = 1.0e-3;    // m / (s^3 * sqrt(Hz))

  // Stationary IMU samples used for initial roll/pitch & gyro bias alignment
  int init_imu_samples = 40;

  // Radar velocity std floor, radar frame [m/s]. Added to the IRLS covariance,
  // which on its own is far smaller than the observed velocity error.
  Eigen::Vector3d radar_vel_std = Eigen::Vector3d(0.10, 0.15, 0.20);

  // Innovation gate on NIS (3 DoF chi-square, 99.9%)
  double nis_gate = 16.27;

  // After this many rejections in a row, velocity uncertainty is inflated
  // so that a diverged filter can accept measurements again
  int max_consecutive_rejections = 20;
};

enum class UpdateResult { kAccepted, kRejected, kRecovering, kInvalid };

class ImuSystemModel;
class RadarVelocityMeasurementModel;

}

namespace kalmanif {
namespace internal {

template <>
struct traits<radom::ImuSystemModel> {
  using State = radom::RioState;
  using Control = radom::ImuControl;
};

template <>
struct traits<radom::RadarVelocityMeasurementModel> {
  using State = radom::RioState;
  using Measurement = radom::RadarMeas;
  static constexpr Invariance invariance = Invariance::Right;
};

}
}

namespace radom {

class ImuSystemModel
    : public kalmanif::SystemModelBase<ImuSystemModel>,
      public kalmanif::LinearizedInvariant<
          kalmanif::SystemModelBase<ImuSystemModel>> {
public:
  using Base = kalmanif::SystemModelBase<ImuSystemModel>;
  using typename Base::State;
  using typename Base::Control;
  using typename Base::Scalar;
  using Base::getCovariance;
  using Base::setCovariance;

  using StateJacobian = kalmanif::Jacobian<State, State>;
  using NoiseJacobian = kalmanif::Jacobian<State, Control>;
  using NoiseCovariance = kalmanif::Covariance<Control>;

  explicit ImuSystemModel(const RioConfig &cfg = RioConfig{});

  // Nominal strap-down kinematics: X_k = f(X_{k-1}, u_k, dt)
  State run(const State &state, const Control &u, Scalar dt) const;

  // Combined propagation + right-invariant Jacobians for kalmanif::IEKF
  State run_linearized_invariant(const State &state,
                                 const Control &u,
                                 StateJacobian &F,
                                 NoiseJacobian &W,
                                 Scalar dt) const;

private:
  RioConfig cfg_;
};

class RadarVelocityMeasurementModel
    : public kalmanif::MeasurementModelBase<RadarVelocityMeasurementModel>,
      public kalmanif::LinearizedInvariant<
          kalmanif::MeasurementModelBase<RadarVelocityMeasurementModel>> {
public:
  using Base = kalmanif::MeasurementModelBase<RadarVelocityMeasurementModel>;
  using typename Base::State;
  using typename Base::Measurement;
  using Base::getCovariance;
  using Base::setCovariance;

  using MeasurementJacobian = kalmanif::Jacobian<Measurement, State>;
  using NoiseJacobian = kalmanif::Jacobian<Measurement, Measurement>;
  using NoiseCovariance = kalmanif::Covariance<Measurement>;

  explicit RadarVelocityMeasurementModel(const RioConfig &cfg = RioConfig{});

  void setObservationContext(const Eigen::Vector3d &gyro_raw,
                             const Eigen::Matrix3d &irls_vel_cov);

  // Predicted radar-frame velocity: h(X) = R_br^T * (R_wb^T * v_w + (w_raw - b_g) x l_br)
  Measurement run(const State &state) const;

  // Combined measurement expectation + right-invariant Jacobians for kalmanif::IEKF
  Measurement run_linearized_invariant(const State &state,
                                       MeasurementJacobian &H,
                                       NoiseJacobian &M) const;

private:
  RioConfig cfg_;
  Eigen::Matrix3d R_rb_;
  Eigen::Matrix3d l_br_skew_;
  Eigen::Matrix3d gyro_cov_discrete_;
  Eigen::Vector3d last_gyro_raw_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d irls_vel_cov_ = Eigen::Matrix3d::Identity() * 1e-3;
};

class RadarInertialFilter {
public:
  using Filter = kalmanif::InvariantExtendedKalmanFilter<RioState>;

  explicit RadarInertialFilter(const RioConfig &cfg = RioConfig{});

  bool initializeFromStaticImu(const std::vector<ImuMeasurement> &imu_batch);
  void propagateImu(const ImuMeasurement &imu, double dt);
  UpdateResult updateRadarVelocity(const Eigen::Vector3d &v_radar,
                                   const Eigen::Matrix3d &irls_vel_cov,
                                   const Eigen::Vector3d &last_gyro_raw);
  // Rotate the heading estimate by `correction` [rad], weighted by `std_dev`
  UpdateResult updateHeading(double correction, double std_dev, double nis_gate);
  void reset();

  // IMU body-frame state accessors
  Eigen::Vector3d position() const {
    return filter_.getState().element<0>().translation();
  }
  Eigen::Quaterniond orientation() const {
    return filter_.getState().element<0>().quat();
  }
  Eigen::Vector3d velocity() const {
    return filter_.getState().element<0>().linearVelocity();
  }
  Eigen::Vector3d gyroBias() const {
    return filter_.getState().element<1>().coeffs();
  }
  Eigen::Vector3d accelBias() const {
    return filter_.getState().element<2>().coeffs();
  }

  // 4D Radar frame pose & velocity accessors (for direct comparison with gt_radar.tum)
  Eigen::Vector3d radarPosition() const {
    return position() + orientation().toRotationMatrix() * cfg_.l_br;
  }
  Eigen::Quaterniond radarOrientation() const {
    return Eigen::Quaterniond(orientation().toRotationMatrix() * cfg_.R_br).normalized();
  }
  Eigen::Vector3d radarLinearVelocity(const Eigen::Vector3d &gyro_raw) const {
    const Eigen::Matrix3d R_wb = orientation().toRotationMatrix();
    const Eigen::Vector3d omega = gyro_raw - gyroBias();
    return cfg_.R_br.transpose() * (R_wb.transpose() * velocity() + omega.cross(cfg_.l_br));
  }

  Eigen::Vector3d radarAngularVelocity(const Eigen::Vector3d &gyro_raw) const {
    return cfg_.R_br.transpose() * (gyro_raw - gyroBias());
  }

  // Covariances of the published radar-frame quantities, ordered as in ROS:
  // [position, rotation] in the world frame and [linear, angular] in the radar frame.
  // covariance() itself is of the right-invariant error and cannot be used directly.
  Eigen::Matrix<double, 6, 6> radarPoseCovariance() const;
  Eigen::Matrix<double, 6, 6> radarTwistCovariance() const;

  const Eigen::Matrix<double, 15, 15> &covariance() const {
    return filter_.getCovariance();
  }
  bool isInitialized() const { return initialized_; }

private:
  static Eigen::Matrix<double, 15, 15> defaultInitialCovariance();

  RioConfig cfg_;
  Filter filter_;
  ImuSystemModel sys_model_;
  RadarVelocityMeasurementModel meas_model_;
  bool initialized_ = false;
  int consecutive_rejections_ = 0;
  std::vector<ImuMeasurement> init_buffer_;
};

}