%% 四旋翼物理参数
% 用于 PID/ADRC/MPC 姿态控制对比实验
% 参考：Crazyflie 2.1 参数

%% 物理参数
params.m = 0.027;           % 质量 [kg]
params.g = 9.81;            % 重力加速度 [m/s^2]
params.Jx = 1.4e-5;         % 滚转惯量 [kg·m^2]
params.Jy = 1.4e-5;         % 俯仰惯量 [kg·m^2]
params.Jz = 2.17e-5;        % 偏航惯量 [kg·m^2]
params.J = diag([params.Jx, params.Jy, params.Jz]);

params.l = 0.046;           % 机臂长度 [m]
params.Ct = 3.1582e-10;     % 推力系数
params.Cd = 7.9379e-12;     % 扭矩系数
params.k_drag = 0.01;       % 线性拖拽系数

%% 电机参数
params.omega_max = 2500;    % 最大转速 [rad/s]
params.omega_min = 0;       % 最小转速 [rad/s]
params.tau_motor = 0.02;    % 电机时间常数 [s]

%% 仿真参数
params.Ts = 0.002;          % 采样周期 [s] (500Hz)
params.Ts_ctrl = 0.005;     % 控制周期 [s] (200Hz)
params.T_sim = 10;          % 仿真时长 [s]

%% 初始状态
params.x0 = zeros(12,1);   % [x,y,z,u,v,w,phi,theta,psi,p,q,r]

%% 风扰参数（Dryden 模型）
params.V_wind = 3;          % 平均风速 [m/s]
params.L_wind = 10;         % 湍流长度尺度 [m]
params.sigma_wind = 1.5;    % 湍流强度 [m/s]

fprintf('参数加载完成: m=%.3fkg, J=[%.2e, %.2e, %.2e]\n', ...
    params.m, params.Jx, params.Jy, params.Jz);
