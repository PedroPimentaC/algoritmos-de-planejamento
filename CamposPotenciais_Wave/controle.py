# controle.py
import math

def norm_angle(a):
    while a > math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

def calcular_forca_atrativa(rx, ry, sub_gx, sub_gy, k_att):
    dist_to_sub = math.hypot(sub_gx - rx, sub_gy - ry)
    if dist_to_sub > 0.01:
        f_x = k_att * ((sub_gx - rx) / dist_to_sub)
        f_y = k_att * ((sub_gy - ry) / dist_to_sub)
    else:
        f_x, f_y = 0.0, 0.0
    return f_x, f_y

def calcular_forca_repulsiva(leituras_sensores, rtheta, angulos_sensores, k_rep, rho_0):
    f_x, f_y = 0.0, 0.0
    for i, dist in enumerate(leituras_sensores):
        if dist is not None and 0.02 < dist < rho_0:
            dist_calc = max(dist, 0.2)
            f_mag = k_rep * ((1.0 / dist_calc) - (1.0 / rho_0)) * (1.0 / (dist_calc**2))
            f_mag = min(f_mag, 5.0) 
            
            global_sensor_angle = norm_angle(rtheta + angulos_sensores[i])
            f_x -= f_mag * math.cos(global_sensor_angle)
            f_y -= f_mag * math.sin(global_sensor_angle)
            
    return f_x, f_y

def calcular_cinematica_pioneer(f_tot_x, f_tot_y, rtheta, kp_omega, v_max, v_rot):
    theta_des = math.atan2(f_tot_y, f_tot_x)
    
    # Erro (Atual - Desejado)
    err_theta = norm_angle(rtheta - theta_des)
    
    omega_calculado = -kp_omega * err_theta
    omega = max(min(omega_calculado, v_rot), -v_rot)
    
    # Projeção da velocidade linear (Cosseno)
    f_mag_total = math.hypot(f_tot_x, f_tot_y)
    v_desejado = f_mag_total * math.cos(err_theta)
    v_atual = max(min(v_desejado, v_max), -v_max)
    
    return v_atual, omega