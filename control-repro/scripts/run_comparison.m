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

% 重新初始化状态
x_pid = zeros(12,1); x_adrc = zeros(12,1); x_mpc = zeros(12,1);
state_pid = init_pid_state(); state_adrc = init_adrc_state(); state_mpc = init_mpc_state();

phi_pid2 = zeros(N_steps,1); phi_adrc2 = zeros(N_steps,1); phi_mpc2 = zeros(N_steps,1);

% 风扰参数：在 t=2s 施加阶跃风扰
wind_time = 2.0;  % 风扰开始时间
wind_force = 0.005; % 风扰力 [N]（相当于约 5m/s 风速）

for k = 1:N_steps
    current_t = t(k);

    % PID 控制
    if mod(k-1, ctrl_step) == 0
        y_pid = [x_pid(7:9); x_pid(10:12); x_pid(3); x_pid(6)];
        [u_pid_now, state_pid] = pid_attitude(ref, y_pid, state_pid, params_pid, Ts_ctrl);
    else
        u_pid_now = u_pid(max(1,k-1),:);
    end

    % ADRC 控制
    if mod(k-1, ctrl_step) == 0
        y_adrc = [x_adrc(7:9); x_adrc(10:12); x_adrc(3); x_adrc(6)];
        [u_adrc_now, state_adrc] = adrc_attitude(ref, y_adrc, state_adrc, params_adrc, Ts_ctrl);
    else
        u_adrc_now = u_adrc(max(1,k-1),:);
    end

    % MPC 控制
    if mod(k-1, ctrl_step) == 0
        y_mpc = [x_mpc(7:9); x_mpc(10:12); x_mpc(3); x_mpc(6)];
        [u_mpc_now, state_mpc] = mpc_attitude(ref, y_mpc, state_mpc, params_mpc, Ts_ctrl);
    else
        u_mpc_now = u_mpc(max(1,k-1),:);
    end

    % 添加风扰（在 x 方向施加力，产生滚转力矩）
    wind_torque = 0;
    if current_t >= wind_time
        wind_torque = wind_force * params.l;  % 风产生的力矩
    end

    u_pid_wind = u_pid_now + [0, wind_torque, 0, 0];
    u_adrc_wind = u_adrc_now + [0, wind_torque, 0, 0];
    u_mpc_wind = u_mpc_now + [0, wind_torque, 0, 0];

    % 状态更新
    [~, x_pid_out] = ode45(@(t,x) quadrotor_eom(t, x, u_pid_wind', params), [0, params.Ts], x_pid);
    x_pid = x_pid_out(end,:)';
    [~, x_adrc_out] = ode45(@(t,x) quadrotor_eom(t, x, u_adrc_wind', params), [0, params.Ts], x_adrc);
    x_adrc = x_adrc_out(end,:)';
    [~, x_mpc_out] = ode45(@(t,x) quadrotor_eom(t, x, u_mpc_wind', params), [0, params.Ts], x_mpc);
    x_mpc = x_mpc_out(end,:)';

    phi_pid2(k) = x_pid(7); phi_adrc2(k) = x_adrc(7); phi_mpc2(k) = x_mpc(7);
end

% 计算风扰指标
max_dev_pid = max(abs(phi_pid2(round(wind_time/Ts_ctrl):end))) * 180/pi;
max_dev_adrc = max(abs(phi_adrc2(round(wind_time/Ts_ctrl):end))) * 180/pi;
max_dev_mpc = max(abs(phi_mpc2(round(wind_time/Ts_ctrl):end))) * 180/pi;

fprintf('风扰抑制指标（最大偏差 [deg]）:\n');
fprintf('  PID:  %.2f\n  ADRC: %.2f\n  MPC:  %.2f\n', max_dev_pid, max_dev_adrc, max_dev_mpc);

% 绘图
figure('Position', [100, 100, 800, 400]);
plot(t, rad2deg(phi_pid2), 'b-', t, rad2deg(phi_adrc2), 'r--', t, rad2deg(phi_mpc2), 'g-.');
hold on; xline(wind_time, 'k:', '风扰开始', 'LineWidth', 1.5);
xlabel('时间 [s]'); ylabel('滚转角 [deg]');
legend('PID', 'ADRC', 'MPC', 'Location', 'best');
title('实验2: 风扰抑制对比（5m/s 阶跃风）'); grid on;
saveas(gcf, fullfile(results_dir, 'exp2_disturbance.png'));
fprintf('实验2完成\n');

%% ========== 实验3: 参数敏感度 ==========
fprintf('\n========== 实验3: 参数敏感度 ==========\n');

% 惯性参数偏移 ±20%
perturbations = [0.8, 1.0, 1.2];  % -20%, nominal, +20%
results_sensitivity = zeros(3, 3);  % [perturbation × algorithm]

for p_idx = 1:length(perturbations)
    pert = perturbations(p_idx);
    params_pert = params;
    params_pert.Jx = params.Jx * pert;
    params_pert.Jy = params.Jy * pert;
    params_pert.Jz = params.Jz * pert;

    % 重新初始化
    x_pid = zeros(12,1); x_adrc = zeros(12,1); x_mpc = zeros(12,1);
    state_pid = init_pid_state(); state_adrc = init_adrc_state(); state_mpc = init_mpc_state();

    phi_pert_pid = zeros(N_steps,1); phi_pert_adrc = zeros(N_steps,1); phi_pert_mpc = zeros(N_steps,1);

    for k = 1:N_steps
        if mod(k-1, ctrl_step) == 0
            y_pid = [x_pid(7:9); x_pid(10:12); x_pid(3); x_pid(6)];
            [u_pid_now, state_pid] = pid_attitude(ref, y_pid, state_pid, params_pid, Ts_ctrl);
            y_adrc = [x_adrc(7:9); x_adrc(10:12); x_adrc(3); x_adrc(6)];
            [u_adrc_now, state_adrc] = adrc_attitude(ref, y_adrc, state_adrc, params_adrc, Ts_ctrl);
            y_mpc = [x_mpc(7:9); x_mpc(10:12); x_mpc(3); x_mpc(6)];
            [u_mpc_now, state_mpc] = mpc_attitude(ref, y_mpc, state_mpc, params_mpc, Ts_ctrl);
        else
            u_pid_now = u_pid(max(1,k-1),:);
            u_adrc_now = u_adrc(max(1,k-1),:);
            u_mpc_now = u_mpc(max(1,k-1),:);
        end

        [~, x_pid_out] = ode45(@(t,x) quadrotor_eom(t, x, u_pid_now', params_pert), [0, params.Ts], x_pid);
        x_pid = x_pid_out(end,:)';
        [~, x_adrc_out] = ode45(@(t,x) quadrotor_eom(t, x, u_adrc_now', params_pert), [0, params.Ts], x_adrc);
        x_adrc = x_adrc_out(end,:)';
        [~, x_mpc_out] = ode45(@(t,x) quadrotor_eom(t, x, u_mpc_now', params_pert), [0, params.Ts], x_mpc);
        x_mpc = x_mpc_out(end,:)';

        phi_pert_pid(k) = x_pid(7); phi_pert_adrc(k) = x_adrc(7); phi_pert_mpc(k) = x_mpc(7);
    end

    % 计算 RMSE
    rmse_pid = sqrt(mean((phi_pert_pid - phi_ref).^2)) * 180/pi;
    rmse_adrc = sqrt(mean((phi_pert_adrc - phi_ref).^2)) * 180/pi;
    rmse_mpc = sqrt(mean((phi_pert_mpc - phi_ref).^2)) * 180/pi;

    results_sensitivity(p_idx, :) = [rmse_pid, rmse_adrc, rmse_mpc];
end

fprintf('参数敏感度（RMSE [deg]）:\n');
fprintf('  %-10s %-10s %-10s %-10s\n', '偏移', 'PID', 'ADRC', 'MPC');
for p_idx = 1:length(perturbations)
    fprintf('  %-10s %-10.2f %-10.2f %-10.2f\n', ...
        sprintf('J×%.1f', perturbations(p_idx)), results_sensitivity(p_idx,1), results_sensitivity(p_idx,2), results_sensitivity(p_idx,3));
end

% 绘图
figure('Position', [100, 100, 600, 400]);
bar(results_sensitivity);
set(gca, 'XTickLabel', {'J×0.8', 'J×1.0', 'J×1.2'});
xlabel('惯性参数偏移'); ylabel('RMSE [deg]');
legend('PID', 'ADRC', 'MPC', 'Location', 'best');
title('实验3: 参数敏感度对比'); grid on;
saveas(gcf, fullfile(results_dir, 'exp3_sensitivity.png'));
fprintf('实验3完成\n');

%% ========== 结果统计 ==========
fprintf('\n========== 结果统计 ==========\n');

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
