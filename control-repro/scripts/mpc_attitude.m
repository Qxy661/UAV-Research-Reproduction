function [u, state] = mpc_attitude(ref, y, state, params, Ts)
% MPC_ATTITUDE MPC 姿态控制器
%   使用滚动优化处理约束
%   输入:
%     ref   - 参考姿态 [phi_ref, theta_ref, psi_ref, z_ref]
%     y     - 当前状态 [phi, theta, psi, p, q, r, z, w]
%     state - 控制器状态
%     params - MPC 参数
%     Ts    - 采样时间
%
%   输出:
%     u     - 控制输入 [f, tau_phi, tau_theta, tau_psi]
%     state - 更新后的控制器状态

    % 解析状态
    phi = y(1);
    theta = y(2);
    psi = y(3);
    p = y(4);
    q = y(5);
    r = y(6);
    z = y(7);
    w = y(8);

    % MPC 参数
    N = params.N;           % 预测步长
    Q = params.Q;           % 状态权重
    R = params.R;           % 输入权重
    dR = params.dR;         % 输入变化率权重

    % 约束
    phi_max = deg2rad(30);  % 最大滚转角
    theta_max = deg2rad(30); % 最大俯仰角
    tau_max = params.l * params.Ct * params.omega_max^2;
    dtau_max = tau_max * 0.5;  % 变化率约束

    % ========== 单通道 MPC（以俯仰为例） ==========
    % 状态空间模型：x = [theta, theta_dot], u = tau_theta
    % A = [1, Ts; 0, 1], B = [0; Ts/Jy]

    Jy = params.Jy;
    A = [1, Ts; 0, 1];
    B = [0; Ts/Jy];

    % 当前状态
    x0 = [theta; q];

    % 参考轨迹
    theta_ref = ref(2);
    x_ref = [theta_ref; 0];

    % 构建预测矩阵
    % X = F*x0 + Phi*U
    F = zeros(2*N, 2);
    Phi = zeros(2*N, N);

    for i = 1:N
        F(2*i-1:2*i, :) = A^i;
        for j = 1:i
            Phi(2*i-1:2*i, j) = A^(i-j) * B;
        end
    end

    % 构建权重矩阵
    Q_tilde = kron(eye(N), Q);
    R_tilde = kron(eye(N), R);

    % 构建输入变化率矩阵
    % ΔU = D * U - d0
    D = eye(N) - diag(ones(N-1,1), -1);
    D(1, :) = 0;  % 第一行不约束

    % QP 目标函数：min 0.5*U'*H*U + f'*U
    H = Phi' * Q_tilde * Phi + R_tilde + dR * (D' * D);
    H = (H + H') / 2;  % 确保对称

    f_qp = Phi' * Q_tilde * (F * x0 - kron(ones(N,1), x_ref));

    % 约束：|theta| <= theta_max, |tau| <= tau_max, |Δtau| <= dtau_max
    % 状态约束转换为输入约束（简化处理）
    A_ineq = [eye(N); -eye(N); D; -D];
    b_ineq = [tau_max * ones(N,1); tau_max * ones(N,1);
              dtau_max * ones(N,1); dtau_max * ones(N,1)];

    % 求解 QP（使用 MATLAB quadprog）
    options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'active-set');

    try
        [U_opt, ~, exitflag] = quadprog(H, f_qp, A_ineq, b_ineq, [], [], [], [], [], options);

        if exitflag ~= 1
            % QP 求解失败，使用无约束解
            U_opt = -H \ f_qp;
        end
    catch
        % quadprog 不可用，使用无约束解
        U_opt = -H \ f_qp;
    end

    % 取第一步控制输入
    tau_theta = U_opt(1);

    % ========== 滚转通道 MPC ==========
    x0_phi = [phi; p];
    phi_ref_val = ref(1);
    x_ref_phi = [phi_ref_val; 0];

    Jx = params.Jx;
    B_phi = [0; Ts/Jx];

    Phi_phi = zeros(2*N, N);
    for i = 1:N
        for j = 1:i
            Phi_phi(2*i-1:2*i, j) = A^(i-j) * B_phi;
        end
    end

    H_phi = Phi_phi' * Q_tilde * Phi_phi + R_tilde + dR * (D' * D);
    H_phi = (H_phi + H_phi') / 2;
    f_phi = Phi_phi' * Q_tilde * (F * x0_phi - kron(ones(N,1), x_ref_phi));

    try
        U_phi = quadprog(H_phi, f_phi, A_ineq, b_ineq, [], [], [], [], [], options);
        if exitflag ~= 1
            U_phi = -H_phi \ f_phi;
        end
    catch
        U_phi = -H_phi \ f_phi;
    end
    tau_phi = U_phi(1);

    % ========== 偏航通道 MPC ==========
    x0_psi = [psi; r];
    psi_ref_val = ref(3);
    x_ref_psi = [psi_ref_val; 0];

    Jz = params.Jz;
    B_psi = [0; Ts/Jz];

    Phi_psi = zeros(2*N, N);
    for i = 1:N
        for j = 1:i
            Phi_psi(2*i-1:2*i, j) = A^(i-j) * B_psi;
        end
    end

    H_psi = Phi_psi' * Q_tilde * Phi_psi + R_tilde + dR * (D' * D);
    H_psi = (H_psi + H_psi') / 2;
    f_psi = Phi_psi' * Q_tilde * (F * x0_psi - kron(ones(N,1), x_ref_psi));

    try
        U_psi = quadprog(H_psi, f_psi, A_ineq, b_ineq, [], [], [], [], [], options);
        if exitflag ~= 1
            U_psi = -H_psi \ f_psi;
        end
    catch
        U_psi = -H_psi \ f_psi;
    end
    tau_psi = U_psi(1);

    % ========== 高度通道（简单 PD） ==========
    z_ref_val = ref(4);
    e_z = z_ref_val - z;
    f_ff = params.m * params.g;
    f = f_ff + params.Kp_z * e_z - params.Kd_z * w;

    % 输出限幅
    f_max = 4 * params.Ct * params.omega_max^2;
    f = max(0, min(f_max, f));
    tau_phi = max(-tau_max, min(tau_max, tau_phi));
    tau_theta = max(-tau_max, min(tau_max, tau_theta));
    tau_psi = max(-tau_max, min(tau_max, tau_psi));

    u = [f; tau_phi; tau_theta; tau_psi];
end
