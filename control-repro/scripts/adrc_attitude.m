function [u, state] = adrc_attitude(ref, y, state, params, Ts)
% ADRC_ATTITUDE ADRC 姿态控制器
%   使用 ESO 观测并补偿总扰动
%   输入:
%     ref   - 参考姿态 [phi_ref, theta_ref, psi_ref, z_ref]
%     y     - 当前状态 [phi, theta, psi, p, q, r, z, w]
%     state - 控制器状态（ESO 状态、积分项）
%     params - ADRC 参数
%     Ts    - 采样时间
%
%   输出:
%     u     - 控制输入 [f, tau_phi, tau_theta, tau_psi]
%     state - 更新后的控制器状态

    % 解析参考
    phi_ref = ref(1);
    theta_ref = ref(2);
    psi_ref = ref(3);
    z_ref = ref(4);

    % 解析当前状态
    phi = y(1);
    theta = y(2);
    psi = y(3);
    p = y(4);
    q = y(5);
    r = y(6);
    z = y(7);
    w = y(8);

    % ADRC 参数
    omega0 = params.omega0;   % ESO 带宽
    omega_c = params.omega_c; % 控制器带宽
    b0_phi = params.b0_phi;   % 对象增益估计（滚转）
    b0_theta = params.b0_theta; % 对象增益估计（俯仰）
    b0_psi = params.b0_psi;   % 对象增益估计（偏航）

    % 带宽参数化
    beta1 = 3 * omega0;
    beta2 = 3 * omega0^2;
    beta3 = omega0^3;

    % ========== ESO 更新（滚转通道） ==========
    e1_phi = state.z1_phi - phi;
    state.z1_phi = state.z1_phi + Ts * (state.z2_phi - beta1 * e1_phi);
    state.z2_phi = state.z2_phi + Ts * (state.z3_phi - beta2 * e1_phi + b0_phi * state.u_phi_prev);
    state.z3_phi = state.z3_phi + Ts * (-beta3 * e1_phi);

    % ========== ESO 更新（俯仰通道） ==========
    e1_theta = state.z1_theta - theta;
    state.z1_theta = state.z1_theta + Ts * (state.z2_theta - beta1 * e1_theta);
    state.z2_theta = state.z2_theta + Ts * (state.z3_theta - beta2 * e1_theta + b0_theta * state.u_theta_prev);
    state.z3_theta = state.z3_theta + Ts * (-beta3 * e1_theta);

    % ========== ESO 更新（偏航通道） ==========
    e1_psi = state.z1_psi - psi;
    state.z1_psi = state.z1_psi + Ts * (state.z2_psi - beta1 * e1_psi);
    state.z2_psi = state.z2_psi + Ts * (state.z3_psi - beta2 * e1_psi + b0_psi * state.u_psi_prev);
    state.z3_psi = state.z3_psi + Ts * (-beta3 * e1_psi);

    % ========== ESO 更新（高度通道） ==========
    e1_z = state.z1_z - z;
    state.z1_z = state.z1_z + Ts * (state.z2_z - beta1 * e1_z);
    state.z2_z = state.z2_z + Ts * (state.z3_z - beta2 * e1_z + params.b0_z * state.u_z_prev);
    state.z3_z = state.z3_z + Ts * (-beta3 * e1_z);

    % ========== 控制律（NLSEF） ==========
    % 滚转通道
    e_phi = phi_ref - state.z1_phi;
    de_phi = 0 - state.z2_phi;  % 假设参考导数为 0
    u0_phi = omega_c^2 * e_phi + 2 * 0.707 * omega_c * de_phi;
    tau_phi = (u0_phi - state.z3_phi) / b0_phi;

    % 俯仰通道
    e_theta = theta_ref - state.z1_theta;
    de_theta = 0 - state.z2_theta;
    u0_theta = omega_c^2 * e_theta + 2 * 0.707 * omega_c * de_theta;
    tau_theta = (u0_theta - state.z3_theta) / b0_theta;

    % 偏航通道
    e_psi = psi_ref - state.z1_psi;
    de_psi = 0 - state.z2_psi;
    u0_psi = omega_c^2 * e_psi + 2 * 0.707 * omega_c * de_psi;
    tau_psi = (u0_psi - state.z3_psi) / b0_psi;

    % 高度通道
    e_z = z_ref - state.z1_z;
    de_z = 0 - state.z2_z;
    f_ff = params.m * params.g;
    u0_z = omega_c^2 * e_z + 2 * 0.707 * omega_c * de_z;
    f = f_ff + (u0_z - state.z3_z) / params.b0_z;

    % 保存上次控制输入（用于 ESO）
    state.u_phi_prev = tau_phi;
    state.u_theta_prev = tau_theta;
    state.u_psi_prev = tau_psi;
    state.u_z_prev = f - f_ff;

    % 输出限幅
    f_max = 4 * params.Ct * params.omega_max^2;
    f = max(0, min(f_max, f));

    tau_max = params.l * params.Ct * params.omega_max^2;
    tau_phi = max(-tau_max, min(tau_max, tau_phi));
    tau_theta = max(-tau_max, min(tau_max, tau_theta));
    tau_psi = max(-tau_max, min(tau_max, tau_psi));

    u = [f; tau_phi; tau_theta; tau_psi];
end
