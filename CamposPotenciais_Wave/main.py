try:
    import sim
except:
    print('Arquivos da API do CoppeliaSim não encontrados. Verifique se o sim.py, simConst.py e a remoteApi.dll estão na mesma pasta.')
    import sys
    sys.exit()

import sys
import time
import math
import numpy as np

import wavefront
import controle


# PARÂMETROS GERAIS

L = 0.331
r = 0.09751
V_MAX = 0.3
V_ROT = 0.8
DT    = 0.05

K_ATT = 0.5
K_REP = 0.02
RHO_0 = 0.5
Kp_omega = 1.2

RESOLUCAO = 10.0 / 50.0  
ORIGEM_X = -5.0
ORIGEM_Y = -5.0

SENSOR_ANGLES = [math.radians(a) for a in [90, 50, 30, 10, -10, -30, -50, -90, -90, -130, -150, -170, 170, 150, 130, 90]]

def set_v_omega(clientID, l_motor, r_motor, v, omega):
    wr = ((2.0 * v) + (omega * L)) / (2.0 * r)
    wl = ((2.0 * v) - (omega * L)) / (2.0 * r)    
    sim.simxSetJointTargetVelocity(clientID, l_motor, wl, sim.simx_opmode_streaming)
    sim.simxSetJointTargetVelocity(clientID, r_motor, wr, sim.simx_opmode_streaming)


# INICIALIZAÇÃO DA SIMULAÇÃO

print('Iniciando o programa principal...')
sim.simxFinish(-1)
clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)

if clientID == -1:
    print('Falha ao ligar ao CoppeliaSim.')
    sys.exit()

print('Conectado ao CoppeliaSim!')

sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
time.sleep(0.5)

robotname = 'Pioneer_p3dx'
_, robotHandle = sim.simxGetObjectHandle(clientID, robotname, sim.simx_opmode_oneshot_wait)
_, l_wheel = sim.simxGetObjectHandle(clientID, robotname + '_leftMotor', sim.simx_opmode_oneshot_wait)
_, r_wheel = sim.simxGetObjectHandle(clientID, robotname + '_rightMotor', sim.simx_opmode_oneshot_wait)
_, goalHandle = sim.simxGetObjectHandle(clientID, 'Goal', sim.simx_opmode_oneshot_wait)

sensors = []
for i in range(1, 17):
    _, s = sim.simxGetObjectHandle(clientID, f'{robotname}_ultrasonicSensor{i}', sim.simx_opmode_oneshot_wait)
    sensors.append(s)
    sim.simxReadProximitySensor(clientID, s, sim.simx_opmode_streaming)
    
sim.simxGetObjectPosition(clientID, robotHandle, -1, sim.simx_opmode_streaming)
sim.simxGetObjectOrientation(clientID, robotHandle, -1, sim.simx_opmode_streaming)
sim.simxGetObjectPosition(clientID, goalHandle, -1, sim.simx_opmode_streaming)
time.sleep(0.5)

# EXTRAÇÃO DO MAPA BRUTO

print("Procurando a câmera no CoppeliaSim...")
res_handle, cam_handle = sim.simxGetObjectHandle(clientID, 'MapaCamera', sim.simx_opmode_oneshot_wait)

if res_handle != sim.simx_return_ok:
    print("ERRO CRÍTICO: Não encontrei a 'MapaCamera'.")
    sim.simxFinish(clientID)
    sys.exit()

print("Extraindo a planta baixa via Vision Sensor...")
sim.simxGetVisionSensorImage(clientID, cam_handle, 0, sim.simx_opmode_streaming)

timeout = time.time() + 5.0
res = -1
while res != sim.simx_return_ok and time.time() < timeout:
    res, resolution, image = sim.simxGetVisionSensorImage(clientID, cam_handle, 0, sim.simx_opmode_buffer)
    time.sleep(0.1)

if res == sim.simx_return_ok:
    img_np = np.array(image).astype(np.uint8)
    img_np.resize([resolution[1], resolution[0], 3])
    img_np = np.flipud(img_np)
    img_gray = np.mean(img_np, axis=2)
    
    MAPA_AMBIENTE = np.where(img_gray < 100, 999, 0)
                
    print(f"Mapa extraído bruto! Matriz: {resolution[0]}x{resolution[1]}")
else:
    print("Timeout ao tentar receber a imagem.")
    sim.simxFinish(clientID)
    sys.exit()

# CÁLCULO DO WAVEFRONT GERAL

_, goal_pos = sim.simxGetObjectPosition(clientID, goalHandle, -1, sim.simx_opmode_buffer)
g_row, g_col = wavefront.world_to_grid(goal_pos[0], goal_pos[1], ORIGEM_X, ORIGEM_Y, RESOLUCAO)

print("Calculando o mapa do Wavefront...")
mapa_wavefront = wavefront.compute_wavefront(MAPA_AMBIENTE, g_row, g_col)
print("Wavefront calculado! Iniciando loop de controle...\n")


# LOOP PRINCIPAL

start_time = time.time()
lastTime = start_time

try:
    while True:
        now = time.time()
        if now - lastTime < DT: continue
        lastTime = now

        # Leitura dos sensores
        _, pos = sim.simxGetObjectPosition(clientID, robotHandle, -1, sim.simx_opmode_buffer)
        _, ori = sim.simxGetObjectOrientation(clientID, robotHandle, -1, sim.simx_opmode_buffer)
        rx, ry, rtheta = pos[0], pos[1], ori[2]
        
        leituras_ultrassom = []
        for s in sensors:
            res_sensor, state, point, _, _ = sim.simxReadProximitySensor(clientID, s, sim.simx_opmode_buffer)
            if state:
                leituras_ultrassom.append(np.linalg.norm(point))
            else:
                leituras_ultrassom.append(None)
        
        # Condição de Chegada
        r_row, r_col = wavefront.world_to_grid(rx, ry, ORIGEM_X, ORIGEM_Y, RESOLUCAO)
        if mapa_wavefront[r_row, r_col] == 0:
            print(f"\nALVO ALCANÇADO! Tempo: {time.time() - start_time:.2f} segundos")
            break

        # Processamento híbrido
        sub_gx, sub_gy = wavefront.get_next_waypoint(r_row, r_col, mapa_wavefront, ORIGEM_X, ORIGEM_Y, RESOLUCAO)
        
        f_att_x, f_att_y = controle.calcular_forca_atrativa(rx, ry, sub_gx, sub_gy, K_ATT)
        f_rep_x, f_rep_y = controle.calcular_forca_repulsiva(leituras_ultrassom, rtheta, SENSOR_ANGLES, K_REP, RHO_0)
        
        f_tot_x = f_att_x + f_rep_x
        f_tot_y = f_att_y + f_rep_y
        
        v_atual, omega = controle.calcular_cinematica_pioneer(f_tot_x, f_tot_y, rtheta, Kp_omega, V_MAX, V_ROT)

        # Ação
        print(f"Navegando... Grid[{r_row}][{r_col}] | Vel: {v_atual:.2f} | Giro: {omega:.2f}", end='\r')
        set_v_omega(clientID, l_wheel, r_wheel, v_atual, omega)

except KeyboardInterrupt:
    print("\n\nSimulação interrompida.")
finally:
    set_v_omega(clientID, l_wheel, r_wheel, 0, 0)
    sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot_wait)
    sim.simxFinish(clientID)
    print("Conexão encerrada.")