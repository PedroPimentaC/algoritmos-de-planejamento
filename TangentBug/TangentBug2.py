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

L = 0.331     # Distância entre-eixos em metros
r = 0.09751   # Raio da roda em metros

GOAL_RANGE     = 0.3
OBSTACLE_RANGE = 0.6
WALL_DIST      = 0.50  # Margem de segurança
V_MAX          = 0.3   # Velocidade linear máxima (m/s)
V_ROT          = 0.5   # Velocidade de giro no próprio eixo (rad/s)
DT             = 0.05

# Ganhos PID - Move to Goal (Controla a rotação para mirar no alvo)
Kp_t = 1.5
Ki_t = 0.01
Kd_t = 0.1
err_sum_theta = 0.0
last_err_theta = 0.0

# Ganhos PID - Wall Follow (Controla a rotação para manter a distância da parede)
Kp_w = 2.0
Ki_w = 0.05
Kd_w = 0.3
err_sum_wall = 0.0
last_err_wall = 0.0


# FUNÇÕES DE APOIO E CINEMÁTICA


def norm_angle(a):
    while a > math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

def distancia(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def set_v_omega(clientID, l_motor, r_motor, v, omega):
    """
    Cinemática Inversa do Acionamento Diferencial
    Recebe v (m/s) e omega (rad/s) e aplica nas rodas
    """
    wr = ((2.0 * v) + (omega * L)) / (2.0 * r)
    wl = ((2.0 * v) - (omega * L)) / (2.0 * r)    
    
    sim.simxSetJointTargetVelocity(clientID, l_motor, wl, sim.simx_opmode_streaming)
    sim.simxSetJointTargetVelocity(clientID, r_motor, wr, sim.simx_opmode_streaming)

def reset_pid():
    global err_sum_theta, last_err_theta, err_sum_wall, last_err_wall
    err_sum_theta = 0.0
    last_err_theta = 0.0
    err_sum_wall = 0.0
    last_err_wall = 0.0


# CONEXÃO E INICIALIZAÇÃO

print('Iniciando o programa...')
sim.simxFinish(-1)
clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)

if clientID != -1:
    print('Conectado ao CoppeliaSim!')
    
    # Handles do Pioneer
    robotname = 'Pioneer_p3dx'
    _, robotHandle = sim.simxGetObjectHandle(clientID, robotname, sim.simx_opmode_oneshot_wait)
    _, l_wheel = sim.simxGetObjectHandle(clientID, robotname + '_leftMotor', sim.simx_opmode_oneshot_wait)
    _, r_wheel = sim.simxGetObjectHandle(clientID, robotname + '_rightMotor', sim.simx_opmode_oneshot_wait)
    _, goalHandle = sim.simxGetObjectHandle(clientID, 'Goal', sim.simx_opmode_oneshot_wait)
    
    # Inicializando Sensores (Pioneer tem 16 sensores, indexados de 1 a 16)
    sensors = []
    for i in range(1, 17):
        _, s = sim.simxGetObjectHandle(clientID, f'{robotname}_ultrasonicSensor{i}', sim.simx_opmode_oneshot_wait)
        sensors.append(s)
        # Primeira leitura para iniciar o streaming
        sim.simxReadProximitySensor(clientID, s, sim.simx_opmode_streaming)
        
    # Primeira leitura de pose para iniciar streaming
    sim.simxGetObjectPosition(clientID, robotHandle, -1, sim.simx_opmode_streaming)
    sim.simxGetObjectOrientation(clientID, robotHandle, -1, sim.simx_opmode_streaming)
    sim.simxGetObjectPosition(clientID, goalHandle, -1, sim.simx_opmode_streaming)
    
    time.sleep(0.5) # Tempo para o streaming preencher os buffers

    # Angulos dos 16 sensores do Pioneer (frente = 0, positivo = esquerda)
    SENSOR_ANGLES = [math.radians(a) for a in [90, 50, 30, 10, -10, -30, -50, -90, -90, -130, -150, -170, 170, 150, 130, 90]]
    FRONT_SENSORS = [2, 3, 4, 5]
    LEFT_SENSORS  = [0, 1, 2]
    RIGHT_SENSORS = [5, 6, 7]

    # Estados do Tangent Bug
    STATE_MOTION = 0
    STATE_FOLLOW = 1
    state = STATE_MOTION
    
    d_followed_min = float('inf')
    hit_point = None
    follow_dir = 1

    print("Iniciando navegação Tangent Bug com PID...")
    
    start_time = time.time() # INÍCIO DO CRONÔMETRO
    lastTime = start_time
    
    try:
        while True:
            now = time.time()
            dt = now - lastTime
            if dt < DT:
                continue
            lastTime = now

            # Leituras Globais
            _, pos = sim.simxGetObjectPosition(clientID, robotHandle, -1, sim.simx_opmode_buffer)
            _, ori = sim.simxGetObjectOrientation(clientID, robotHandle, -1, sim.simx_opmode_buffer)
            _, goal_pos = sim.simxGetObjectPosition(clientID, goalHandle, -1, sim.simx_opmode_buffer)
            
            rx, ry, rtheta = pos[0], pos[1], ori[2]
            gx, gy = goal_pos[0], goal_pos[1]
            d_goal = distancia(rx, ry, gx, gy)

            if d_goal < GOAL_RANGE:
                tempo_total = time.time() - start_time # CÁLCULO DO TEMPO TOTAL
                print("\nALVO ALCANÇADO!")
                print(f"Tempo gasto até o objetivo: {tempo_total:.2f} segundos")
                break

            # Processamento dos Sensores
            raw = []
            for i, s in enumerate(sensors):
                res, detectionState, detectedPoint, _, _ = sim.simxReadProximitySensor(clientID, s, sim.simx_opmode_buffer)
                if detectionState:
                    dist = np.linalg.norm(detectedPoint)
                    raw.append((i, SENSOR_ANGLES[i], dist))
                else:
                    raw.append((i, SENSOR_ANGLES[i], float('inf')))

            front_dist = min([raw[i][2] for i in FRONT_SENSORS])
            blocked_angles = [angle for i, angle, d in raw if d < OBSTACLE_RANGE]
            obstacle_detected = len(blocked_angles) > 0
            angle_to_goal = norm_angle(math.atan2(gy - ry, gx - rx) - rtheta)

            
            # ESTADO: MOTION TO GOAL
            
            if state == STATE_MOTION:
                if not obstacle_detected:
                    # Controle PID de Orientação
                    err = angle_to_goal
                    err_sum_theta += err * dt
                    d_err = (err - last_err_theta) / dt
                    last_err_theta = err
                    
                    omega = (Kp_t * err) + (Ki_t * err_sum_theta) + (Kd_t * d_err)
                    
                    # Suaviza a velocidade linear se o erro angular for muito grande
                    v_atual = V_MAX * max(0.1, 1.0 - (abs(err) / math.pi))
                    set_v_omega(clientID, l_wheel, r_wheel, v_atual, omega)
                    
                else:
                    blocked_min = min(blocked_angles)
                    blocked_max = max(blocked_angles)
                    candidate_left  = blocked_max + math.radians(15)
                    candidate_right = blocked_min - math.radians(15)

                    err_left  = abs(norm_angle(candidate_left - angle_to_goal))
                    err_right = abs(norm_angle(candidate_right - angle_to_goal))
                    follow_dir = 1 if err_left < err_right else -1

                    if angle_to_goal < blocked_min or angle_to_goal > blocked_max:
                        # Rotação PID se o alvo estiver fora da zona bloqueada
                        err = angle_to_goal
                        err_sum_theta += err * dt
                        d_err = (err - last_err_theta) / dt
                        last_err_theta = err
                        omega = (Kp_t * err) + (Ki_t * err_sum_theta) + (Kd_t * d_err)
                        set_v_omega(clientID, l_wheel, r_wheel, V_MAX * 0.5, omega)
                    else:
                        hit_point = (rx, ry)
                        d_followed_min = d_goal
                        dir_str = 'esquerda' if follow_dir == 1 else 'direita'
                        print(f"Obstáculo detectado! Iniciando contorno pela {dir_str}.")
                        reset_pid()
                        state = STATE_FOLLOW

            
            # ESTADO: WALL FOLLOWING
            
            elif state == STATE_FOLLOW:
                if d_goal < d_followed_min:
                    d_followed_min = d_goal

                side_dist = min([raw[i][2] for i in (LEFT_SENSORS if follow_dir == 1 else RIGHT_SENSORS)])
                d_hit_to_goal = distancia(hit_point[0], hit_point[1], gx, gy)

                # Condição de saída: caminho livre para o alvo e mais perto que no hit point
                if front_dist > OBSTACLE_RANGE and side_dist > OBSTACLE_RANGE and d_goal < d_hit_to_goal:
                    print("Caminho desobstruído. Retomando Motion-to-Goal.")
                    hit_point = None
                    reset_pid()
                    state = STATE_MOTION
                else:
                    if front_dist < OBSTACLE_RANGE:
                        # Obstáculo frontal crítico: Gira no próprio eixo
                        omega = -V_ROT if follow_dir == 1 else V_ROT
                        set_v_omega(clientID, l_wheel, r_wheel, 0.0, omega)
                        reset_pid() # Zera o PID lateral pois a manobra foi interrompida
                    else:
                        # Saturar a distância lida pelo PID para evitar problemas com as quinas
                        limite_fuga = WALL_DIST + 0.15
                        side_dist_pid = min(side_dist, limite_fuga)

                        # Controle PID Lateral usando o valor saturado
                        err = side_dist_pid - WALL_DIST
                        err_sum_wall += err * dt
                        d_err = (err - last_err_wall) / dt
                        last_err_wall = err

                        correction = (Kp_w * err) + (Ki_w * err_sum_wall) + (Kd_w * d_err)
                        
                        # Se sigo pela esquerda (follow_dir=1), para me aproximar da parede (err > 0), giro para a esquerda (omega positivo)
                        omega = correction if follow_dir == 1 else -correction
                        
                        # Limita a velocidade angular do contorno
                        omega = max(min(omega, V_ROT), -V_ROT)
                        
                        # Nas quinas, diminui um pouco a velocidade linear para dar tempo de girar
                        velocidade_frente = V_MAX * 0.7 if side_dist <= WALL_DIST + 0.05 else V_MAX * 0.4
                        
                        set_v_omega(clientID, l_wheel, r_wheel, velocidade_frente, omega)

    except KeyboardInterrupt:
        print("\nSimulação interrompida pelo utilizador.")
    finally:
        sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot_wait)
        set_v_omega(clientID, l_wheel, r_wheel, 0, 0)
        sim.simxFinish(clientID)
        print("Simulação interrompida e conexão encerrada.")

else:
    print('Falha ao ligar ao CoppeliaSim.')