try:
    import sim
except:
    print('Arquivos da API do CoppeliaSim não encontrados.')
    import sys
    sys.exit()

import time
import math
import numpy as np

# PARÂMETROS DO ROBÔ E DA CINEMÁTICA

L = 0.331  
r = 0.09751

V_MAX = 0.3 
DT    = 0.05


# PARÂMETROS DOS CAMPOS POTENCIAIS

K_ATT = 0.5
K_REP = 0.01
RHO_0 = 1.5
GOAL_RANGE = 0.3
Kp_omega = 1.0


# FUNÇÕES DE APOIO

def norm_angle(a):
    while a > math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

def distancia(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def set_v_omega(clientID, l_motor, r_motor, v, omega):
    wr = ((2.0 * v) + (omega * L)) / (2.0 * r)
    wl = ((2.0 * v) - (omega * L)) / (2.0 * r)
    sim.simxSetJointTargetVelocity(clientID, l_motor, wl, sim.simx_opmode_streaming)
    sim.simxSetJointTargetVelocity(clientID, r_motor, wr, sim.simx_opmode_streaming)


# CONEXÃO E INICIALIZAÇÃO

print('Iniciando o programa...')
sim.simxFinish(-1)
clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)

if clientID != -1:
    print('Conectado ao CoppeliaSim! Iniciando Campos Potenciais...')
    
    # Handles do Pioneer e Objetivo
    robotname = 'Pioneer_p3dx'
    _, robotHandle = sim.simxGetObjectHandle(clientID, robotname, sim.simx_opmode_oneshot_wait)
    _, l_wheel = sim.simxGetObjectHandle(clientID, robotname + '_leftMotor', sim.simx_opmode_oneshot_wait)
    _, r_wheel = sim.simxGetObjectHandle(clientID, robotname + '_rightMotor', sim.simx_opmode_oneshot_wait)
    _, goalHandle = sim.simxGetObjectHandle(clientID, 'Goal', sim.simx_opmode_oneshot_wait)
    
    # Inicializando os 16 Sensores
    sensors = []
    for i in range(1, 17):
        _, s = sim.simxGetObjectHandle(clientID, f'{robotname}_ultrasonicSensor{i}', sim.simx_opmode_oneshot_wait)
        sensors.append(s)
        sim.simxReadProximitySensor(clientID, s, sim.simx_opmode_streaming)
        
    sim.simxGetObjectPosition(clientID, robotHandle, -1, sim.simx_opmode_streaming)
    sim.simxGetObjectOrientation(clientID, robotHandle, -1, sim.simx_opmode_streaming)
    sim.simxGetObjectPosition(clientID, goalHandle, -1, sim.simx_opmode_streaming)
    time.sleep(0.5)

    # Ângulos fixos dos sensores do Pioneer (frente = 0, positivo = esquerda)
    SENSOR_ANGLES = [math.radians(a) for a in [90, 50, 30, 10, -10, -30, -50, -90, -90, -130, -150, -170, 170, 150, 130, 90]]

    start_time = time.time()
    lastTime = start_time
    
    try:
        while True:
            now = time.time()
            dt = now - lastTime
            if dt < DT:
                continue
            lastTime = now

            # Leituras de Posição
            _, pos = sim.simxGetObjectPosition(clientID, robotHandle, -1, sim.simx_opmode_buffer)
            _, ori = sim.simxGetObjectOrientation(clientID, robotHandle, -1, sim.simx_opmode_buffer)
            _, goal_pos = sim.simxGetObjectPosition(clientID, goalHandle, -1, sim.simx_opmode_buffer)
            
            rx, ry, rtheta = pos[0], pos[1], ori[2]
            gx, gy = goal_pos[0], goal_pos[1]
            
            d_goal = distancia(rx, ry, gx, gy)

            if d_goal < GOAL_RANGE:
                print(f"\nALVO ALCANÇADO! Tempo: {time.time() - start_time:.2f} segundos")
                break

            # CÁLCULO DAS FORÇAS (VETORES)
            
            # Força Atrativa
            F_att_x = K_ATT * (gx - rx)
            F_att_y = K_ATT * (gy - ry)
            
            # Força Repulsiva
            F_rep_x = 0.0
            F_rep_y = 0.0

            for i, s in enumerate(sensors):
                res, detectionState, detectedPoint, _, _ = sim.simxReadProximitySensor(clientID, s, sim.simx_opmode_buffer)
                if detectionState:
                    dist = np.linalg.norm(detectedPoint)
                    
                    # Se o obstáculo está dentro do raio de influência
                    if 0.01 < dist < RHO_0: 
                        # Cálculo da magnitude da força de repulsão
                        F_mag = K_REP * ((1.0 / dist) - (1.0 / RHO_0)) * (1.0 / (dist**2))
                        
                        # Ângulo global em que o sensor está apontando
                        global_sensor_angle = norm_angle(rtheta + SENSOR_ANGLES[i])
                        
                        # O vetor de repulsão tem sentido OPOSTO ao obstáculo
                        F_rep_x -= F_mag * math.cos(global_sensor_angle)
                        F_rep_y -= F_mag * math.sin(global_sensor_angle)

            # Força Resultante
            F_tot_x = F_att_x + F_rep_x
            F_tot_y = F_att_y + F_rep_y

            # CINEMÁTICA E CONTROLE
            
            # Descobre para onde o vetor resultante está mandando o robô ir
            theta_des = math.atan2(F_tot_y, F_tot_x)
            
            # Erro de orientação atual do robô para a direção desejada
            err_theta = norm_angle(rtheta - theta_des)
            
            # Velocidade angular proporcional ao erro
            omega = (-Kp_omega) * err_theta
            
            # Velocidade linear é máxima se ele estiver alinhado, e zera se precisar girar muito (> 90 graus)
            F_mag_total = math.hypot(F_tot_x, F_tot_y)
            v_desejada = F_mag_total * math.cos(err_theta)
            v_atual = max(min(v_desejada, V_MAX), -V_MAX)
            
            # Aplica nos motores
            set_v_omega(clientID, l_wheel, r_wheel, v_atual, omega)

    except KeyboardInterrupt:
        print("\nSimulação interrompida pelo utilizador.")
    finally:
        sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot_wait)
        set_v_omega(clientID, l_wheel, r_wheel, 0, 0)
        sim.simxFinish(clientID)
        print("Conexão encerrada.")

else:
    print('Falha ao ligar ao CoppeliaSim.')