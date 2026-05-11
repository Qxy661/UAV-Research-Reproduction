function dx = quadrotor_eom(t, x, u, params)
% QUADROTOR_EOM 四旋翼12状态动力学方程
%   状态: x = [x,y,z,u,v,w,phi,theta,psi,p,q,r]
%   输入: u = [f, tau_phi, tau_theta, tau_psi]
%
%   用于 ode45 求解器

    % 解析状态
    pos = x(1:3);       % [x, y, z]
    vel = x(4:6);       % [u, v, w]
    euler = x(7:9);     % [phi, theta, psi]
    omega = x(10:12);   % [p, q, r]

    phi = euler(1);
    theta = euler(2);
    psi = euler(3);
    p = omega(1);
    q = omega(2);
    r = omega(3);

    % 解析输入
    f = u(1);           % 总推力
    tau = u(2:4);       % [tau_phi, tau_theta, tau_psi]

    % 物理参数
    m = params.m;
    g = params.g;
    J = params.J;
    k_drag = params.k_drag;

    % 旋转矩阵 R (body to world)
    R = [cos(psi)*cos(theta) - sin(phi)*sin(psi)*sin(theta), ...
         -cos(phi)*sin(psi), ...
         cos(psi)*sin(theta) + cos(theta)*sin(phi)*sin(psi);
         cos(theta)*sin(psi) + cos(phi)*cos(psi)*sin(theta), ...
         cos(phi)*cos(psi), ...
         sin(psi)*sin(theta) - cos(psi)*cos(theta)*sin(phi);
         -sin(theta), ...
         sin(phi)*cos(theta), ...
         cos(phi)*cos(theta)];

    % 平移动力学
    % m * [u_dot, v_dot, w_dot] = R * [0; 0; f] + [0; 0; -m*g] - k_drag * [u; v; w]
    accel = (R * [0; 0; f] + [0; 0; -m*g] - k_drag * vel) / m;

    % 欧拉角速率
    % [phi_dot, theta_dot, psi_dot] = W * [p, q, r]
    W = [1, sin(phi)*tan(theta), cos(phi)*tan(theta);
         0, cos(phi), -sin(phi);
         0, sin(phi)/cos(theta), cos(phi)/cos(theta)];
    euler_dot = W * omega;

    % 旋转动力学
    % J * omega_dot = tau - omega x (J * omega)
    omega_dot = J \ (tau - cross(omega, J * omega));

    % 状态导数
    dx = zeros(12, 1);
    dx(1:3) = vel;
    dx(4:6) = accel;
    dx(7:9) = euler_dot;
    dx(10:12) = omega_dot;
end
