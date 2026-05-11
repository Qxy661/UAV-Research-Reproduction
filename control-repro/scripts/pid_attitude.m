function [u, state] = pid_attitude(ref, y, state, params, Ts)
% PID_ATTITUDE PID 姿态控制器
%   输入:
%     ref   - 参考姿态 [phi_ref, theta_ref, psi_ref, z_ref]
%     y     - 当前状态 [phi, theta, psi, p, q, r, z, w]
%     state - 控制器状态（积分项、上次误差）
%     params - PID 参数
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

    % PID 参数（外环：姿态角）
    Kp_att = params.Kp_att;
    Ki_att = params.Ki_att;
    Kd_att = params.Kd_att;

    % PID 参数（内环：角速率）
    Kp_rate = params.Kp_rate;
    Ki_rate = params.Ki_rate;
    Kd_rate = params.Kd_rate;

    % PID 参数（高度环）
    Kp_z = params.Kp_z;
    Ki_z = params.Ki_z;
    Kd_z = params.Kd_z;

    % ========== 外环：姿态角 → 角速率参考 ==========
    e_phi = phi_ref - phi;
    e_theta = theta_ref - theta;
    e_psi = psi_ref - psi;

    % 积分项（带抗饱和）
    state.int_phi = state.int_phi + e_phi * Ts;
    state.int_theta = state.int_theta + e_theta * Ts;
    state.int_psi = state.int_psi + e_psi * Ts;

    % 积分限幅
    int_max = 0.5;
    state.int_phi = max(-int_max, min(int_max, state.int_phi));
    state.int_theta = max(-int_max, min(int_max, state.int_theta));
    state.int_psi = max(-int_max, min(int_max, state.int_psi));

    % 角速率参考
    p_ref = Kp_att * e_phi + Ki_att * state.int_phi + Kd_att * (e_phi - state.e_phi_prev) / Ts;
    q_ref = Kp_att * e_theta + Ki_att * state.int_theta + Kd_att * (e_theta - state.e_theta_prev) / Ts;
    r_ref = Kp_att * e_psi + Ki_att * state.int_psi + Kd_att * (e_psi - state.e_psi_prev) / Ts;

    % 保存误差
    state.e_phi_prev = e_phi;
    state.e_theta_prev = e_theta;
    state.e_psi_prev = e_psi;

    % ========== 内环：角速率 → 力矩 ==========
    e_p = p_ref - p;
    e_q = q_ref - q;
    e_r = r_ref - r;

    % 积分项
    state.int_p = state.int_p + e_p * Ts;
    state.int_q = state.int_q + e_q * Ts;
    state.int_r = state.int_r + e_r * Ts;

    % 积分限幅
    state.int_p = max(-int_max, min(int_max, state.int_p));
    state.int_q = max(-int_max, min(int_max, state.int_q));
    state.int_r = max(-int_max, min(int_max, state.int_r));

    % 力矩输出
    tau_phi = Kp_rate * e_p + Ki_rate * state.int_p + Kd_rate * (e_p - state.e_p_prev) / Ts;
    tau_theta = Kp_rate * e_q + Ki_rate * state.int_q + Kd_rate * (e_q - state.e_q_prev) / Ts;
    tau_psi = Kp_rate * e_r + Ki_rate * state.int_r + Kd_rate * (e_r - state.e_r_prev) / Ts;

    % 保存误差
    state.e_p_prev = e_p;
    state.e_q_prev = e_q;
    state.e_r_prev = e_r;

    % ========== 高度环 ==========
    e_z = z_ref - z;
    state.int_z = state.int_z + e_z * Ts;
    state.int_z = max(-int_max, min(int_max, state.int_z));

    % 推力输出（前馈 + 反馈）
    f_ff = params.m * params.g;  % 悬停推力前馈
    f_fb = Kp_z * e_z + Ki_z * state.int_z + Kd_z * (0 - w);  % 反馈
    f = f_ff + f_fb;

    % 输出限幅
    f_max = 4 * params.Ct * params.omega_max^2;
    f = max(0, min(f_max, f));

    tau_max = params.l * params.Ct * params.omega_max^2;
    tau_phi = max(-tau_max, min(tau_max, tau_phi));
    tau_theta = max(-tau_max, min(tau_max, tau_theta));
    tau_psi = max(-tau_max, min(tau_max, tau_psi));

    u = [f; tau_phi; tau_theta; tau_psi];
end
