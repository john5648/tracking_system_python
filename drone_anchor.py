# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2017-2018 Bitcraze AB
#
#  Crazyflie Nano Quadcopter Client
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA  02110-1301, USA.
"""
Original code from AutonomousSequence.py from bitcraze/crazyflie-lib-python
This is a code that four drones following one drone

The layout of the positions:
    URI2              URI1


            TAG(CENTER)         URI5        +------> X
                      


    URI3               URI4
"""
import time

import cflib.crtp
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.swarm import CachedCfFactory
# from cflib.crazyflie.swarm import Swarm
from cflib.crazyflie.syncLogger import SyncLogger
import numpy as np
import pandas as pd
import Swarm_change
from math import cos, sin, pi

# Change uris and sequences according to your setup
URI1 = 'radio://0/120/2M/E7E7E7E7EB'
URI2 = 'radio://0/80/2M/E7E7E7E7EC'
URI3 = 'radio://0/125/2M/E7E7E7E7ED'
URI4 = 'radio://0/125/2M/E7E7E7E7EE'
URI5 = 'radio://0/100/2M/E7E7E7E7EF'

# URI6 and URI7 is a dummy URI
URI6 = 'radio://0/100/2M/E7E7E7E71A'
URI7 = 'radio://0/100/2M/E7E7E7E71B'

Kalmanpred = [-10,-10]
############## parameters to change##################
center = [3,3]
tag_height = 0.3
anchor_height = 1.7
#length of shape
length = 1
polygon = 5
command = 'none'
######################################################
#    x   y   z  time arrangment of x arrangement of y
# arrangement of x and y is used after anchor drones take off
con_vec = [0,0,0,0,0]
for i in range(0,5):
    con_vec[i] = [length*cos(pi/180*360/polygon*(i+1)), length*sin(pi/180*360/polygon*(i+1))]
    if abs(con_vec[i][0])<0.01:
        con_vec[i][0]=0.0
    if abs(con_vec[i][1])<0.01:
        con_vec[i][1]=0.0

#   [(x   y   z), control vector, array number]
sequence1 = [
    [center[0] + con_vec[0][0], center[1] + con_vec[0][1], anchor_height], con_vec[0], 0
]
sequence2 = [
    [center[0] + con_vec[1][0], center[1] + con_vec[1][1], anchor_height], con_vec[1], 1
]
sequence3 = [
    [center[0] + con_vec[2][0], center[1] + con_vec[2][1], anchor_height], con_vec[2], 2
]
sequence4 = [
    [center[0] + con_vec[3][0], center[1] + con_vec[3][1], anchor_height], con_vec[3], 3
]
sequence5 = [
    [center[0] + con_vec[4][0], center[1] + con_vec[4][1], anchor_height], con_vec[4], 4
]

seq_args = {
    URI1: [sequence1],
    URI2: [sequence2],
    URI3: [sequence3],
    URI4: [sequence4],  
    URI5: [sequence5],     
    URI6: [sequence5],  
    URI7: [sequence5],  
}

# List of URIs, comment the one you do not want to fly
uris = {
    URI1,
    URI2,
    URI3,
    URI4,
    URI5,
    URI6,
    URI7
}

a=[]

'''
-------------------Kalman setting-------------------
'''
dt=0.1
A=np.array([[1,dt,0,0],[0,1,0,0],[0,0,1,dt],[0,0,0,1]])
Hk=np.array([[1,0,0,0],[0,0,1,0]])
Q=np.eye(4)/400
R=10*np.eye(2)
xk=np.array([[center[0]],[0],[center[1]],[0]])
P=100*np.eye(4)
'''
------------------Kalman parameter-------------------
xk : position state (np.array([x],[x'],[y],[y']))
xp : predicted state
K : kalman gain (K down -> smooth, which weights more to past)
P : state
Q : Q up   -> K up, Q down -> K down
R : R down -> K up, R up   -> K down
'''

def TrackKalman(z):
    global A, Hk, Q, R, xk, P
    # kalman
    xp=A@xk
    Pp=A@P@A.T+Q
    K=Pp@Hk.T@np.linalg.inv(Hk@Pp@Hk.T+R)
    xk=xp+K@(z-Hk@xp)
    P=Pp-K@Hk@Pp
    #
    point=np.array([[xk[0]],[xk[2]]])
    return point

def toafunc(H,r):
    # H
    dummy=np.zeros(H.shape)
    dummy=dummy+H[0]
    toaH=H-dummy
    # K**2
    toaK2=(sum((H*H).T)).reshape(H.shape[0],1)
    # r
    toar=r
    toar2=(sum((toar*toar).T)).reshape(toar.shape[0],1)
    # b
    toaB=1/2*(toaK2-toaK2[0]-toar2+toar[0]**2)
    # row delete
    toaH=toaH[1:,:]
    toaB=toaB[1:,:]
    # toa prediction
    TOApred=np.linalg.inv(toaH.T@toaH)@toaH.T@toaB
    return TOApred

class TOC:
    def __init__(self, cf):
        self._cf = cf
        self._link = cf.link_uri

        self.log_conf2 = LogConfig(name='LoPoTab0', period_in_ms=100)
        self.log_conf2.add_variable('ranging.distance0', 'float')

        self.log_conf = LogConfig(name='Position', period_in_ms=100)
        self.log_conf.add_variable('kalman.stateX', 'float')
        self.log_conf.add_variable('kalman.stateY', 'float')
        self.log_conf.add_variable('kalman.stateZ', 'float')

        self._cf.log.add_config(self.log_conf2)
        self.log_conf2.data_received_cb.add_callback(self.ranging_callback)
        self.log_conf2.start()

        self._cf.log.add_config(self.log_conf)
        self.log_conf.data_received_cb.add_callback(self.position_callback)
        self.log_conf.start()
  
    def ranging_callback(self, timestamp, data, logconf):
        self.range1 = data['ranging.distance0']
        
    def position_callback(self, timestamp, data, logconf):
        self.x = data['kalman.stateX']
        self.y = data['kalman.stateY']
        self.z = data['kalman.stateZ']

def wait_for_param_download(scf):    
    cf = scf.cf
    if cf.link_uri[-2:] >='AA':
        while not scf.cf.param.is_updated:
            time.sleep(1.0)   
        print('Parameters downloaded for', scf.cf.link_uri)

def log_download(scf):
    global drone1, drone2, drone3, drone4, drone5

    cf = scf.cf    
    if cf.link_uri ==URI1:
        drone1=TOC(cf)
    elif cf.link_uri ==URI2:
        drone2=TOC(cf)
    elif cf.link_uri ==URI3:
        drone3=TOC(cf)
    elif cf.link_uri ==URI4:
        drone4=TOC(cf)
    elif cf.link_uri ==URI5:
        drone5=TOC(cf)

def take_off(cf, position):
    take_off_time = 2.0
    sleep_time = 0.1
    steps = int(take_off_time / sleep_time)
    vz = position[2] / take_off_time

    for i in range(steps):
        cf.commander.send_velocity_world_setpoint(0, 0, vz, 0)
        time.sleep(sleep_time)

def land(cf, position):
    landing_time = 2.0
    sleep_time = 0.1
    steps = int(landing_time / sleep_time)
    vz = -position[2] / landing_time

    for _ in range(steps):
        cf.commander.send_velocity_world_setpoint(0, 0, vz, 0)
        time.sleep(sleep_time)

    cf.commander.send_stop_setpoint()
    # Make sure that the last packet leaves before the link is closed
    # since the message queue is not flushed before closing
    time.sleep(0.1)

def run_sequence(scf, sequence):
    global a, TOAloc, Kalmanpred, command
    try:
        cf = scf.cf
        if cf.link_uri == URI6:
            time.sleep(2)
            while (command != 'end'):           
                H = [[drone1.x, drone1.y], [drone2.x, drone2.y], [drone3.x, drone3.y], [drone4.x, drone4.y], [drone5.x, drone5.y]]
                r = [np.sqrt(np.abs((drone1.range1*0.001)**2-(drone1.z-tag_height)**2)),np.sqrt(np.abs((drone2.range1*0.001)**2-(drone2.z-tag_height)**2)),np.sqrt(np.abs((drone3.range1*0.001)**2-(drone3.z-tag_height)**2)),np.sqrt(np.abs((drone4.range1*0.001)**2-(drone4.z-tag_height)**2)),np.sqrt(np.abs((drone5.range1*0.001)**2-(drone5.z-tag_height)**2))]

                HH = np.array(H)
                rr = np.array(r)

                TOAloc = toafunc(HH,rr.reshape(5,1)).reshape(1,2)
                Kalmanpred=TrackKalman(TOAloc.reshape(2,1)).reshape(2)

                a=a+[[ drone1.range1, drone2.range1, drone3.range1, drone4.range1, drone5.range1, drone1.x, drone1.y, drone1.z, drone2.x, drone2.y, drone2.z, drone3.x, drone3.y, drone3.z, drone4.x, drone4.y, drone4.z, drone5.x, drone5.y, drone5.z,TOAloc[0][0], TOAloc[0][1], Kalmanpred[0], Kalmanpred[1]]]
                if command =='':
                    a = a + [[0 for i in range(24)]]
                    command ='none'
                time.sleep(0.1)

            ddd=pd.DataFrame(a)
            ddd.to_csv('data.csv')

        elif cf.link_uri == URI7: 
            while (command != 'end'):
                command = input()
        else:
            take_off(cf, sequence[0])
            print('Setting position {}'.format(sequence[0]))
            start_time = time.time()
            while (command != 'end'):
                if time.time() < start_time + 3:
                    cf.commander.send_position_setpoint(sequence[0][0],sequence[0][1],sequence[0][2], 0)
                else:
                    cf.commander.send_position_setpoint(Kalmanpred[0]+sequence[1][0],Kalmanpred[1]+sequence[1][1],sequence[0][2], 0)
                # time.sleep(0.05)
            land(cf, sequence[0])

    except Exception as e:
        print(e)

# this class is to share data with GUI code
class trans_self:
    def __init__(self):
        self.xyz = 0

    def trans_var(self):
        global drone1, drone2, drone3, drone4, drone5, Kalmanpred
        self.xyz = [[drone1.x, drone1.y, drone1.z],[drone2.x, drone2.y, drone2.z], [drone3.x, drone3.y, drone3.z],[drone4.x, drone4.y, drone4.z], [drone5.x, drone5.y, drone5.z], [Kalmanpred[0],Kalmanpred[1],0]]

if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    cflib.crtp.init_drivers(enable_debug_driver=False)

    factory = CachedCfFactory(rw_cache='./cache')
    with Swarm_change.Swarm(uris, factory=factory) as swarm:
        # If the copters are started in their correct positions this is
        # probably not needed. The Kalman filter will have time to converge
        # any way since it takes a while to start them all up and connect. We
        # keep the code here to illustrate how to do it.
        # swarm.parallel(reset_estimator)

        # The current values of all parameters are downloaded as a part of the
        # connections sequence. Since we have 10 copters this is clogging up
        # communication and we have to wait for it to finish before we start
        # flying.
        swarm.parallel(log_download)
        print('Waiting for parameters to be downloaded...')
        swarm.parallel(wait_for_param_download)
        input('Enter to start')
        swarm.parallel(run_sequence, args_dict=seq_args)
        ddd=pd.DataFrame(a)
        ddd.to_csv('data.csv')