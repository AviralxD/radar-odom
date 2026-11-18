FROM osrf/ros:humble-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

ARG USERNAME=queen
ARG UID=1000
ARG GID=1000
ARG WS_NAME=ros_ws
ARG ENTRYPOINT=entrypoint.sh
ARG TMUXP_CONF=tmuxp.yaml

ENV USERNAME=${USERNAME}
ENV WS_NAME=${WS_NAME}

# Configure locale, Qt5/X11 GUI support, and HeRCULES ROS 2 dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    sudo \
    tmux \
    qtbase5-dev \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    ros-humble-novatel-gps-msgs \
    ros-humble-rosbag2-storage-mcap \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN if getent passwd ${UID} >/dev/null 2>&1; then userdel -f $(getent passwd ${UID} | cut -d: -f1); fi \
    && if ! getent group ${GID} >/dev/null 2>&1; then groupadd --gid ${GID} ${USERNAME}; fi \
    && useradd -s /bin/bash --uid ${UID} --gid ${GID} -m ${USERNAME} \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && usermod -aG video ${USERNAME}

USER ${USERNAME}
WORKDIR /home/${USERNAME}/${WS_NAME}

COPY --chown=${UID}:${GID} ./${TMUXP_CONF} /home/${USERNAME}/${WS_NAME}/tmuxp.yaml
COPY --chown=${UID}:${GID} ./colcon.meta /home/${USERNAME}/${WS_NAME}/colcon.meta
COPY --chown=${UID}:${GID} ./${ENTRYPOINT} /home/${USERNAME}/${WS_NAME}/entrypoint.sh

RUN chmod +x ./entrypoint.sh

RUN echo "source /opt/ros/humble/setup.bash" >> /home/${USERNAME}/.bashrc && \
    echo "[ -f /home/${USERNAME}/${WS_NAME}/install/local_setup.bash ] && source /home/${USERNAME}/${WS_NAME}/install/local_setup.bash" >> /home/${USERNAME}/.bashrc && \
    echo "export DISABLE_AUTO_TITLE='true'" >> /home/${USERNAME}/.bashrc

CMD ["sleep", "infinity"]