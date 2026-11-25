#include "radom/rio_iekf.hh"

#include <algorithm>
#include <cmath>

namespace radom {

namespace {

inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v) {
  Eigen::Matrix3d S;
  S <<     0.0, -v.z(),  v.y(),
        v.z(),     0.0, -v.x(),
       -v.y(),  v.x(),     0.0;
  return S;
}

}

ImuSystemModel::ImuSystemModel(const RioConfig &cfg) : cfg_(cfg) {
  NoiseCovariance Qc = NoiseCovariance::Zero();
  Qc.block<3, 3>(0, 0) =
      Eigen::Matrix3d::Identity() * (cfg_.gyro_noise_std * cfg_.gyro_noise_std);
  Qc.block<3, 3>(3, 3) =
      Eigen::Matrix3d::Identity() * (cfg_.accel_noise_std * cfg_.accel_noise_std);
  Qc.block<3, 3>(6, 6) =
      Eigen::Matrix3d::Identity() * (cfg_.gyro_bias_rw_std * cfg_.gyro_bias_rw_std);
  Qc.block<3, 3>(9, 9) =
      Eigen::Matrix3d::Identity() * (cfg_.accel_bias_rw_std * cfg_.accel_bias_rw_std);
  this->setCovariance(Qc);
}

ImuSystemModel::State ImuSystemModel::run(
    const State &state,
    const Control &u,
    Scalar dt) const {
  const auto &X = state.element<0>();
  const Eigen::Vector3d b_g = state.element<1>().coeffs();
  const Eigen::Vector3d b_a = state.element<2>().coeffs();

  const Eigen::Matrix3d R_wb = X.rotation();
  const Eigen::Vector3d v_w = X.linearVelocity();
  const Eigen::Vector3d p_w = X.translation();

  const Eigen::Vector3d gyro_unbiased = u.head<3>() - b_g;
  const Eigen::Vector3d accel_unbiased = u.segment<3>(3) - b_a;

  // Exact SO(3) exponential integration over dt
  const Eigen::Matrix3d dR = manif::SO3Tangentd(gyro_unbiased * dt).exp().rotation();
  const Eigen::Matrix3d R_next = R_wb * dR;

  const Eigen::Vector3d acc_world = R_wb * accel_unbiased + cfg_.gravity;
  const Eigen::Vector3d v_next = v_w + acc_world * dt;
  const Eigen::Vector3d p_next = p_w + v_w * dt + 0.5 * acc_world * (dt * dt);

  const manif::SE_2_3d X_next(p_next, Eigen::Quaterniond(R_next).normalized(), v_next);
  return State(X_next, manif::R3d(b_g), manif::R3d(b_a));
}

ImuSystemModel::State ImuSystemModel::run_linearized_invariant(
    const State &state,
    const Control &u,
    StateJacobian &F,
    NoiseJacobian &W,
    Scalar dt) const {
  const auto &X = state.element<0>();
  const Eigen::Matrix3d R_wb = X.rotation();
  const Eigen::Vector3d v_w = X.linearVelocity();
  const Eigen::Vector3d p_w = X.translation();

  // Continuous-time right-invariant error dynamics F_c (15x15)
  // Error ordering: [dp(0..2), dtheta(3..5), dv(6..8), db_g(9..11), db_a(12..14)]
  StateJacobian Fc = StateJacobian::Zero();
  Fc.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();
  Fc.block<3, 3>(0, 9) = -skewSymmetric(p_w) * R_wb;
  Fc.block<3, 3>(3, 9) = -R_wb;
  Fc.block<3, 3>(6, 3) = skewSymmetric(cfg_.gravity);
  Fc.block<3, 3>(6, 9) = -skewSymmetric(v_w) * R_wb;
  Fc.block<3, 3>(6, 12) = -R_wb;

  // Second-order discrete transition Phi = I + F_c * dt + 0.5 * F_c^2 * dt^2
  F = StateJacobian::Identity() + Fc * dt + 0.5 * (Fc * Fc) * (dt * dt);

  // Continuous-time noise input matrix G_c (15x12)
  // Scaled by sqrt(dt) so W * Q_c * W^T = G_c * Q_c * G_c^T * dt
  NoiseJacobian Gc = NoiseJacobian::Zero();
  Gc.block<3, 3>(0, 0) = -skewSymmetric(p_w) * R_wb;
  Gc.block<3, 3>(3, 0) = -R_wb;
  Gc.block<3, 3>(6, 0) = -skewSymmetric(v_w) * R_wb;
  Gc.block<3, 3>(6, 3) = -R_wb;
  Gc.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();
  Gc.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity();

  // Apply the discrete transition to the continuous noise channels. This carries
  // acceleration noise into velocity and position covariance with the correct dt/dt^2 scaling.
  W = F * Gc * std::sqrt(std::max(dt, 1e-9));

  return run(state, u, dt);
}

RadarVelocityMeasurementModel::RadarVelocityMeasurementModel(const RioConfig &cfg)
    : cfg_(cfg),
      R_rb_(cfg.R_br.transpose()),
      l_br_skew_(skewSymmetric(cfg.l_br)) {
  // Approximate discrete gyro variance (~100 Hz IMU rate) for lever-arm noise propagation
  const double sigma_g2 = (cfg_.gyro_noise_std * cfg_.gyro_noise_std) * 100.0;
  gyro_cov_discrete_ = Eigen::Matrix3d::Identity() * sigma_g2;

  const Eigen::Matrix3d J_gyro = R_rb_ * l_br_skew_;
  const Eigen::Matrix3d floor = cfg_.radar_vel_std.cwiseAbs2().asDiagonal();
  this->setCovariance(irls_vel_cov_ + floor + J_gyro * gyro_cov_discrete_ * J_gyro.transpose());
}

void RadarVelocityMeasurementModel::setObservationContext(
    const Eigen::Vector3d &gyro_raw,
    const Eigen::Matrix3d &irls_vel_cov) {
  last_gyro_raw_ = gyro_raw;
  irls_vel_cov_ = irls_vel_cov;

  const Eigen::Matrix3d J_gyro = R_rb_ * l_br_skew_;
  const Eigen::Matrix3d floor = cfg_.radar_vel_std.cwiseAbs2().asDiagonal();
  this->setCovariance(irls_vel_cov_ + floor + J_gyro * gyro_cov_discrete_ * J_gyro.transpose());
}

RadarVelocityMeasurementModel::Measurement
RadarVelocityMeasurementModel::run(const State &state) const {
  const auto &X = state.element<0>();
  const Eigen::Vector3d b_g = state.element<1>().coeffs();

  const Eigen::Matrix3d R_wb = X.rotation();
  const Eigen::Vector3d v_w = X.linearVelocity();

  const Eigen::Vector3d omega_body = last_gyro_raw_ - b_g;
  const Eigen::Vector3d v_radar_imu_frame =
      R_wb.transpose() * v_w + omega_body.cross(cfg_.l_br);

  return R_rb_ * v_radar_imu_frame;
}

RadarVelocityMeasurementModel::Measurement
RadarVelocityMeasurementModel::run_linearized_invariant(
    const State &state,
    MeasurementJacobian &H,
    NoiseJacobian &M) const {
  const Eigen::Matrix3d R_wb = state.element<0>().rotation();

  // Exact right-invariant cancellation leaves zero blocks for dp (0..2) and dtheta (3..5)
H = MeasurementJacobian::Zero();
  H.block<3, 3>(0, 6) = -R_rb_ * R_wb.transpose();
  H.block<3, 3>(0, 9) = -R_rb_ * l_br_skew_;

  M = NoiseJacobian::Identity();

  return run(state);
}

Eigen::Matrix<double, 15, 15> RadarInertialFilter::defaultInitialCovariance() {
  Eigen::Matrix<double, 15, 15> P = Eigen::Matrix<double, 15, 15>::Identity();
  P.block<3, 3>(0, 0) *= 1e-4;    // position [m^2]
  P.block<3, 3>(3, 3) *= 1e-3;    // orientation [rad^2]
  P.block<3, 3>(6, 6) *= 1e-2;    // velocity [(m/s)^2]
  P.block<3, 3>(9, 9) *= 1e-4;    // gyro bias [(rad/s)^2]
  P.block<3, 3>(12, 12) *= 1e-3;  // accel bias [(m/s^2)^2]
  return P;
}

RadarInertialFilter::RadarInertialFilter(const RioConfig &cfg)
    : cfg_(cfg),
      filter_(RioState::Identity(), defaultInitialCovariance()),
      sys_model_(cfg),
      meas_model_(cfg) {}

void RadarInertialFilter::reset() {
  filter_.setState(RioState::Identity());
  filter_.setCovariance(defaultInitialCovariance());
  init_buffer_.clear();
  initialized_ = false;
  consecutive_rejections_ = 0;
}

bool RadarInertialFilter::initializeFromStaticImu(
    const std::vector<ImuMeasurement> &imu_batch) {
  if (initialized_) return true;

  for (const auto &imu : imu_batch) {
    init_buffer_.push_back(imu);
    if (static_cast<int>(init_buffer_.size()) >= cfg_.init_imu_samples) {
      break;
    }
  }

  if (static_cast<int>(init_buffer_.size()) < cfg_.init_imu_samples) {
    return false;
  }

  Eigen::Vector3d mean_gyro = Eigen::Vector3d::Zero();
  Eigen::Vector3d mean_accel = Eigen::Vector3d::Zero();
  for (const auto &m : init_buffer_) {
    mean_gyro += m.angular_vel;
    mean_accel += m.linear_accel;
  }
  const double inv_n = 1.0 / static_cast<double>(init_buffer_.size());
  mean_gyro *= inv_n;
  mean_accel *= inv_n;

  // Align initial roll & pitch so R_wb * mean_accel aligns with -gravity (+Z in ENU), yaw = 0
  const Eigen::Vector3d g_up = -cfg_.gravity.normalized();
  const Eigen::Vector3d a_dir = mean_accel.normalized();
  Eigen::Quaterniond q_init = Eigen::Quaterniond::FromTwoVectors(a_dir, g_up);

  // Zero out any yaw induced by FromTwoVectors so initial heading is strictly 0 rad
  const Eigen::Vector3d fwd_world = q_init * Eigen::Vector3d::UnitX();
  const double yaw_offset = std::atan2(fwd_world.y(), fwd_world.x());
  const Eigen::Quaterniond q_unyaw(
      Eigen::AngleAxisd(-yaw_offset, Eigen::Vector3d::UnitZ()));
  q_init = (q_unyaw * q_init).normalized();

  // Residual accelerometer bias along gravity direction
  const Eigen::Vector3d b_a_init =
      mean_accel - q_init.toRotationMatrix().transpose() * (-cfg_.gravity);

  const manif::SE_2_3d X_init(
      Eigen::Vector3d::Zero(), q_init, Eigen::Vector3d::Zero());
  filter_.setState(
      RioState(X_init, manif::R3d(mean_gyro), manif::R3d(b_a_init)));
  filter_.setCovariance(defaultInitialCovariance());

  initialized_ = true;
  return true;
}

void RadarInertialFilter::propagateImu(const ImuMeasurement &imu, double dt) {
  if (!initialized_ || dt <= 0.0 || dt > 0.5) return;

  ImuControl u = ImuControl::Zero();
  u.head<3>() = imu.angular_vel;
  u.segment<3>(3) = imu.linear_accel;

  filter_.propagate(sys_model_, u, dt);
}

UpdateResult RadarInertialFilter::updateRadarVelocity(
    const Eigen::Vector3d &v_radar,
    const Eigen::Matrix3d &irls_vel_cov,
    const Eigen::Vector3d &last_gyro_raw) {
  if (!initialized_) return UpdateResult::kInvalid;
  if (!v_radar.allFinite() || !irls_vel_cov.allFinite() || v_radar.norm() > 40.0) {
    return UpdateResult::kInvalid;
  }

  meas_model_.setObservationContext(last_gyro_raw, irls_vel_cov);

  RadarVelocityMeasurementModel::MeasurementJacobian H;
  RadarVelocityMeasurementModel::NoiseJacobian M;
  const Eigen::Vector3d e =
      meas_model_.run_linearized_invariant(filter_.getState(), H, M);
  const Eigen::Vector3d innov = v_radar - e;
  const Eigen::Matrix3d S =
      H * filter_.getCovariance() * H.transpose() + meas_model_.getCovariance();

  Eigen::LDLT<Eigen::Matrix3d> s_ldlt(S);
  if (s_ldlt.info() != Eigen::Success ||
      s_ldlt.vectorD().minCoeff() <= 1e-12) {
    return UpdateResult::kInvalid;
  }
  const double nis = innov.dot(s_ldlt.solve(innov));
  if (!std::isfinite(nis)) {
    return UpdateResult::kInvalid;
  }
  if (nis > cfg_.nis_gate) {
    if (++consecutive_rejections_ < cfg_.max_consecutive_rejections) {
      return UpdateResult::kRejected;
    }
    // The filter, not the radar, is most likely wrong: widen the velocity
    // uncertainty each scan until a measurement passes the gate
    Eigen::Matrix<double, 15, 15> P = filter_.getCovariance();
    P.block<3, 3>(6, 6) += Eigen::Matrix3d::Identity();
    filter_.setCovariance(P);
    return UpdateResult::kRecovering;
  }

  consecutive_rejections_ = 0;
  filter_.update(meas_model_, v_radar);
  return UpdateResult::kAccepted;
}

Eigen::Matrix<double, 6, 6> RadarInertialFilter::radarPoseCovariance() const {
  // Right-invariant error: dp_radar = dp - [p_radar]x dtheta, dtheta_world = dtheta
  Eigen::Matrix<double, 6, 15> J = Eigen::Matrix<double, 6, 15>::Zero();
  J.block<3, 3>(0, 0).setIdentity();
  J.block<3, 3>(0, 3) = -skewSymmetric(radarPosition());
  J.block<3, 3>(3, 3).setIdentity();
  return J * filter_.getCovariance() * J.transpose();
}

Eigen::Matrix<double, 6, 6> RadarInertialFilter::radarTwistCovariance() const {
  const Eigen::Matrix3d R_rb = cfg_.R_br.transpose();
  const Eigen::Matrix3d R_bw = orientation().toRotationMatrix().transpose();

  // Linear velocity depends on dv and db_g, angular velocity on db_g only
  Eigen::Matrix<double, 6, 15> J = Eigen::Matrix<double, 6, 15>::Zero();
  J.block<3, 3>(0, 6) = R_rb * R_bw;
  J.block<3, 3>(0, 9) = R_rb * skewSymmetric(cfg_.l_br);
  J.block<3, 3>(3, 9) = -R_rb;

  // Gyro white noise enters both through the instantaneous reading
  Eigen::Matrix<double, 6, 3> G;
  G.topRows<3>() = -R_rb * skewSymmetric(cfg_.l_br);
  G.bottomRows<3>() = R_rb;
  const double sigma_g2 = cfg_.gyro_noise_std * cfg_.gyro_noise_std * 100.0;

  return J * filter_.getCovariance() * J.transpose() + sigma_g2 * G * G.transpose();
}

}
