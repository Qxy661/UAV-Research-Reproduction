%% run_comparison.m — PID vs ADRC vs MPC 姿态控制对比实验
% 实验1: 阶跃响应
% 实验2: 风扰抑制
% 实验3: 参数敏感度
%
% 用法: 直接运行此脚本

clear; clc; close all;
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
results_dir = fullfile(script_dir, '..', 'results');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end

%% 加载参数
params = struct();
params.m = 0.027; params.g = 9.81;
params.Jx = 1.4e-5; params.Jy = 1.4e-5; params.Jz = 2.17e-5;
params.J = diag([params.Jx, params.Jy, params.Jz]);
params.l = 0.046; params.Ct = 3.1582e-10; params.Cd = 7.9379e-12;
params.k_drag = 0.01;
params.omega_max = 2500; params.omega_min = 0;
params.Ts = 0.002; Ts_ctrl = 0.005;

%% ========== 实验1: 阶跃响应 ==========
fprintf('========== 实验1: 阶跃响应 ==========\n');

% 目标姿态：滚转 10°, 俯仰 10°, 偏航 0°
phi_ref = deg2rad(10);
theta_ref = deg2rad(10);
psi_ref = 0;
z_ref = 1.0;
ref = [phi_ref, theta_ref, psi_ref, z_ref];

% 初始化控制器状态
state_pid = init_pid_state();
state_adrc = init_adrc_state();
state_mpc = init_mpc_state();

% PID 参数
params_pid = params;
params_pid.Kp_att = 6.0; params_pid.Ki_att = 0.1; params_pid.Kd_att = 0.3;
params_pid.Kp_rate = 0.015; params_pid.Ki_rate = 0.001; params_pid.Kd_rate = 0.0001;
params_pid.Kp_z = 10.0; params_pid.Ki_z = 0.5; params_pid.Kd_z = 5.0;

% ADRC 参数
params_adrc = params;
params_adrc.omega0 = 50;      % ESO 带宽
params_adrc.omega_c = 15;     % 控制器带宽
params_adrc.b0_phi = 1/params.Jx;
params_adrc.b0_theta = 1/params.Jy;
params_adrc.b0_psi = 1/params.Jz;
params_adrc.b0_z = 1/params.m;

% MPC 参数
params_mpc = params;
params_mpc.N = 20;
params_mpc.Q = [10, 0; 0, 1];
params_mpc.R = 0.1;
params_mpc.dR = 0.01;
params_mpc.Kp_z = 10.0;
params_mpc.Kd_z = 5.0;

% 仿真时间
T_sim = 5;
t = 0:params.Ts:T_sim;
N_steps = length(t);

% 存储结果
phi_pid = zeros(N_steps, 1); phi_adrc = zeros(N_steps, 1); phi_mpc = zeros(N_steps, 1);
theta_pid = zeros(N_steps, 1); theta_adrc = zeros(N_steps, 1); theta_mpc = zeros(N_steps, 1);
u_pid = zeros(N_steps, 4); u_adrc = zeros(N_steps, 4); u_mpc = zeros(N_steps, 4);

% 初始状态
x_pid = zeros(12, 1);
x_adrc = zeros(12, 1);
x_mpc = zeros(12, 1);

% 运行仿真
ctrl_step = round(Ts_ctrl / params.Ts);
for k = 1:N_steps
    % PID 控制
    if mod(k-1, ctrl_step) == 0
        y_pid = [x_pid(7:9); x_pid(10:12); x_pid(3); x_pid(6)];
        [u_pid(k,:), state_pid] = pid_attitude(ref, y_pid, state_pid, params_pid, Ts_ctrl);
    else
        u_pid(k,:) = u_pid(k-1,:);
    end

    % ADRC 控制
    if mod(k-1, ctrl_step) == 0
        y_adrc = [x_adrc(7:9); x_adrc(10:12); x_adrc(3); x_adrc(6)];
        [u_adrc(k,:), state_adrc] = adrc_attitude(ref, y_adrc, state_adrc, params_adrc, Ts_ctrl);
    else
        u_adrc(k,:) = u_adrc(k-1,:);
    end

    % MPC 控制
    if mod(k-1, ctrl_step) == 0
        y_mpc = [x_mpc(7:9); x_mpc(10:12); x_mpc(3); x_mpc(6)];
        [u_mpc(k,:), state_mpc] = mpc_attitude(ref, y_mpc, state_mpc, params_mpc, Ts_ctrl);
    else
        u_mpc(k,:) = u_mpc(k-1,:);
    end

    % 状态更新（使用 ode45）
    [~, x_pid_out] = ode45(@(t,x) quadrotor_eom(t, x, u_pid(k,:)', params), ...
                           [0, params.Ts], x_pid);
    x_pid = x_pid_out(end,:)';

    [~, x_adrc_out] = ode45(@(t,x) quadrotor_eom(t, x, u_adrc(k,:)', params), ...
                            [0, params.Ts], x_adrc);
    x_adrc = x_adrc_out(end,:)';

    [~, x_mpc_out] = ode45(@(t,x) quadrotor_eom(t, x, u_mpc(k,:)', params), ...
                           [0, params.Ts], x_mpc);
    x_mpc = x_mpc_out(end,:)';

    % 记录
    phi_pid(k) = x_pid(7); phi_adrc(k) = x_adrc(7); phi_mpc(k) = x_mpc(7);
    theta_pid(k) = x_pid(8); theta_adrc(k) = x_adrc(8); theta_mpc(k) = x_mpc(8);
end

% 绘图
figure('Position', [100, 100, 1200, 800]);

subplot(2,2,1);
plot(t, rad2deg(phi_pid), 'b-', t, rad2deg(phi_adrc), 'r--', t, rad2deg(phi_mpc), 'g-.');
hold on; yline(rad2deg(phi_ref), 'k:', 'LineWidth', 1.5);
xlabel('时间 [s]'); ylabel('滚转角 [deg]');
legend('PID', 'ADRC', 'MPC', '参考', 'Location', 'best');
title('实验1: 滚转角阶跃响应'); grid on;

subplot(2,2,2);
plot(t, rad2deg(theta_pid), 'b-', t, rad2deg(theta_adrc), 'r--', t, rad2deg(theta_mpc), 'g-.');
hold on; yline(rad2deg(theta_ref), 'k:', 'LineWidth', 1.5);
xlabel('时间 [s]'); ylabel('俯仰角 [deg]');
legend('PID', 'ADRC', 'MPC', '参考', 'Location', 'best');
title('实验1: 俯仰角阶跃响应'); grid on;

subplot(2,2,3);
plot(t, u_pid(:,2), 'b-', t, u_adrc(:,2), 'r--', t, u_mpc(:,2), 'g-.');
xlabel('时间 [s]'); ylabel('滚转力矩 [N·m]');
legend('PID', 'ADRC', 'MPC', 'Location', 'best');
title('控制输入（滚转力矩）'); grid on;

subplot(2,2,4);
plot(t, u_pid(:,1), 'b-', t, u_adrc(:,1), 'r--', t, u_mpc(:,1), 'g-.');
xlabel('时间 [s]'); ylabel('总推力 [N]');
legend('PID', 'ADRC', 'MPC', 'Location', 'best');
title('控制输入（总推力）'); grid on;

sgtitle('实验1: 阶跃响应对比', 'FontSize', 14);
saveas(gcf, fullfile(results_dir, 'exp1_step_response.png'));
fprintf('实验1完成，结果已保存\n');

%% ========== 实验2: 风扰抑制 ==========
fprintf('\n========== 实验2: 风扰抑制 ==========\n');

% 在 t=2s 施加 5m/s 阶跃风扰
% 复用实验1的代码，但在动力学中加入风扰

% ... (类似结构，风扰通过修改 quadrotor_eom 的 k_drag 项实现)

%% ========== 实验3: 参数敏感度 ==========
fprintf('\n========== 实验3: 参数敏感度 ==========\n');

% 惯性参数偏移 ±20%
% 对每个控制器分别测试，计算性能退化率

% ... (类似结构，修改 params.Jx, params.Jy, params.Jz)

%% ========== 结果统计 ==========
fprintf('\n========== 结果统计 ==========\n');

% 计算性能指标
metrics_exp1 = compute_metrics(t, phi_pid, phi_adrc, phi_mpc, phi_ref);
fprintf('实验1 - 阶跃响应指标:\n');
disp(metrics_exp1);

%% 辅助函数
function state = init_pid_state()
    state.int_phi = 0; state.int_theta = 0; state.int_psi = 0;
    state.int_p = 0; state.int_q = 0; state.int_r = 0;
    state.int_z = 0;
    state.e_phi_prev = 0; state.e_theta_prev = 0; state.e_psi_prev = 0;
    state.e_p_prev = 0; state.e_q_prev = 0; state.e_r_prev = 0;
end

function state = init_adrc_state()
    state.z1_phi = 0; state.z2_phi = 0; state.z3_phi = 0;
    state.z1_theta = 0; state.z2_theta = 0; state.z3_theta = 0;
    state.z1_psi = 0; state.z2_psi = 0; state.z3_psi = 0;
    state.z1_z = 0; state.z2_z = 0; state.z3_z = 0;
    state.u_phi_prev = 0; state.u_theta_prev = 0;
    state.u_psi_prev = 0; state.u_z_prev = 0;
end

function state = init_mpc_state()
    state = struct();  % MPC 无状态
end

function metrics = compute_metrics(t, phi_pid, phi_adrc, phi_mpc, phi_ref)
    % 上升时间（首次达到 90% 目标）
    threshold = 0.9 * phi_ref;
    idx_pid = find(phi_pid >= threshold, 1, 'first');
    idx_adrc = find(phi_adrc >= threshold, 1, 'first');
    idx_mpc = find(phi_mpc >= threshold, 1, 'first');

    tr_pid = t(idx_pid) * 1000;  % ms
    tr_adrc = t(idx_adrc) * 1000;
    tr_mpc = t(idx_mpc) * 1000;

    % 超调量
    os_pid = (max(phi_pid) - phi_ref) / phi_ref * 100;
    os_adrc = (max(phi_adrc) - phi_ref) / phi_ref * 100;
    os_mpc = (max(phi_mpc) - phi_ref) / phi_ref * 100;

    % 稳态误差
    ss_pid = abs(mean(phi_pid(end-100:end)) - phi_ref) / phi_ref * 100;
    ss_adrc = abs(mean(phi_adrc(end-100:end)) - phi_ref) / phi_ref * 100;
    ss_mpc = abs(mean(phi_mpc(end-100:end)) - phi_ref) / phi_ref * 100;

    metrics = table(...
        [tr_pid; tr_adrc; tr_mpc], ...
        [os_pid; os_adrc; os_mpc], ...
        [ss_pid; ss_adrc; ss_mpc], ...
        'VariableNames', {'上升时间_ms', '超调量_pct', '稳态误差_pct'}, ...
        'RowNames', {'PID', 'ADRC', 'MPC'});
end
