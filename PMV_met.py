import cv2, serial, shutil
import detect, detect_person
import os, csv, time
import pandas as pd
from pathlib import Path
from result_util import *
from distutils.dir_util import copy_tree  # when the directory already exists
import numpy as np
import io, socket, struct
from PIL import Image
import datetime as dt
import pythermalcomfort as comfort
import pymysql
import ast

# ★★★★★★★ 이름바꾸기!!!!!! ★★★★★ subject 번호 입력하기 (남자는 _M, 여자는 _F)
subject = 'sub01_M'
control = 'PMV_DCL_control'

saving_path = "/home/ejjchl77/Desktop/EJ_PMV/Mockup_MET_heating/" + subject+"/"+control+"/"

# 아두이노 포트
ir_tcp = '/dev/ttyACM0'              # ------> Room2 적외선, 전류 센서 포트 번호

# 초기변수
system_now = 0
control_start_time = 0
saving_time = 0
not_occu = 0
ac_mode=2
next_setTA = 22
################################################### CLO 모델 정보 #####################################################

def personModel(img_path):
    img_file = os.path.basename(img_path)
    jpg_name, ext = os.path.splitext(img_file)
    start_time = dt.datetime.now()
    
    save_path_per = Path("/home/ejjchl77/Desktop/EJ_PMV/Mockup_MET_heating/"+subject+"/"+control+"/person/")
    ######detect 파일 구분#######
    # detect.py 파일 person 용
    # source: save_path_per
    %run detect_person.py --save-txt --save-conf --save-crop --weights yolov5l.pt --img 416 --device cpu --conf 0.7 --classes 0 --source $img_path --savepath $save_path_per
    person_path = str(save_path_per)+'/crops/person/' + img_file
    
    return person_path

def cloModel(img_path, person_path):

    img_file = os.path.basename(img_path)
    jpg_name, ext = os.path.splitext(img_file)
    start_time = dt.datetime.now()

    ######이미지 저장 폴더 위치 설정#######
    # save_path_per: person detected path img) '/person/corps/person', txt) 'person/labels'
    # save_path_clo: clothing category estimate path img) 'clo_result', txt) 'clo_result/labels'

    save_path_per = Path('/home/ejjchl77/Desktop/EJ_PMV/Mockup_MET_heating/' + subject+'/'+control+'/person/')  
    save_path_clo = os.path.dirname(save_path_per)
    #person_path=str(save_path_per)+'/crops/person/'+ img_file

    pred = []

    # person not detected
    if not os.path.exists(person_path):
        pass

    else:
        ######detect 파일 구분#######
        # detect.py 파일 clo 용
        # source: save_path_clo
        # v1 mode: yolov5l_0614
        %run detect_clo.py --save-txt --save-conf --weights ./runs/train/cloModel_v3_0115/weights/best.pt --img 416 --conf 0.5 --device cpu --source $person_path --savepath $save_path_clo
        clo_path = str(save_path_clo)
        label_path = (clo_path+'/clo_result/labels/'+jpg_name+'.txt')

        if not os.path.exists(label_path): pass
        else:
            # label info
            clo_category = {}
            df = pd.read_csv(label_path, delimiter=' ', header=None)
            start = df.columns[0]
            end = df.columns[-1]
            framename = jpg_name
            clo_category[framename] = []

            for num in range(len(df.index)):
                clo_category[framename].append([df[start][num], df[end][num]])

            pred = select_best_conf_box_real_time(clo_category[framename])

    print(dt.datetime.now()-start_time)

    return pred, jpg_name

def db_upload(sql, val):
    conn = pymysql.connect(host='192.168.0.2',
               user='pbcl',
               password='pbcl7896',
               db='testbed',
               charset='utf8')

    cursor = conn.cursor()

    cursor.execute(sql, val)

    conn.commit()

def db_download(sql, col):
    #데이터 베이스 접속
    conn = pymysql.connect(
        user='pbcl',
        passwd = 'pbcl7896',
        host = '192.168.0.2',
        db = 'testbed',
        charset='utf8'
    )

    #커서 지정
    cursor = conn.cursor()
    cursor.execute(sql)
    cursor.close()

    # 최근 5개 데이터 불러오기 
    result = cursor.fetchall()
    result = pd.DataFrame(result)
    result.columns = col
    return result

date = dt.datetime.today()
now = dt.datetime.now()
now = now.strftime("%Y-%m-%d-%H-%M-%S")

s_raspi_1_1 = socket.socket()
s_raspi_1_2 = socket.socket()

host = '192.168.0.2'  # ip of host
port_raspi_1_1 = 9001
port_raspi_1_2 = 9002
print("waiting")
s_raspi_1_1.bind((host, port_raspi_1_1))
s_raspi_1_2.bind((host, port_raspi_1_2))
print("waiting")
s_raspi_1_1.listen(5)
s_raspi_1_2.listen(5)

while True:
    
    while True:
    
        while True:
            # camera saving
            connection_raspi_1_1 = s_raspi_1_1.accept()[0].makefile('rb')
            connection_raspi_1_2 = s_raspi_1_2.accept()[0].makefile('rb')
            print("get")
        
            camera_start_time = dt.datetime.now()
            camera_start_time = camera_start_time.strftime("%Y-%m-%d-%H-%M-%S")
            img_path = saving_path+'image/'
            os.makedirs(img_path, exist_ok=True)
        
            image_len_raspi_1_1 = struct.unpack('<L', connection_raspi_1_1.read(struct.calcsize('<L')))[0]
            image_len_raspi_1_2 = struct.unpack('<L', connection_raspi_1_2.read(struct.calcsize('<L')))[0]
            image_stream_raspi_1_1 = io.BytesIO()
            image_stream_raspi_1_2 = io.BytesIO()
            image_stream_raspi_1_1.write(connection_raspi_1_1.read(image_len_raspi_1_1))
            image_stream_raspi_1_2.write(connection_raspi_1_2.read(image_len_raspi_1_2))
            image_stream_raspi_1_1.seek(0)
            image_stream_raspi_1_2.seek(0)
            image_raspi_1_1 = Image.open(image_stream_raspi_1_1)
            image_raspi_1_2 = Image.open(image_stream_raspi_1_2)
            image_raspi_1_1.save(img_path + camera_start_time + '_R1_01_image.jpg', 'JPEG')
            image_raspi_1_2.save(img_path + camera_start_time + '_R1_02_image.jpg', 'JPEG')
            print('Image is verified')
            R1_path = os.path.dirname(img_path)
            R1_file = sorted(os.listdir(R1_path))
            new_file = R1_file[-2:]
            print(new_file)
            
            connection_raspi_1_1.close()
            connection_raspi_1_2.close()
        
            # clo 산출 모델 평가
            if not new_file:
                pass
        
            else:
                img_path = [R1_path+'/' + s for s in new_file if camera_start_time in s]
                preds = []
                cnt = 0
        
                for img in img_path:
                    path_dir = os.path.dirname(img)
                    path_img = os.path.basename(img)
        
                    person_img_path = personModel(img)
                    print(person_img_path)
        
                    if not os.path.isfile(person_img_path):  # 사람이 인식안됏을때
                        os.remove(img)
                        os.remove(os.path.dirname(path_dir)+'/person/'+path_img)
                        pp = ['none']
                        preds.append(pp)
                        print("\n산출모델: 사람인식 안됨")
        
                    else:
                        pred, jpg_name = cloModel(img, person_img_path)
                        pred_index = [jpg_name]
                        for i in pred:
                            if not i:
                                pass
                            else:
                                pred_index.append(i[0])
                        preds.append(pred_index)
        
                        os.remove(os.path.dirname(path_dir)+'/person/'+path_img)
                        print("\n산출모델: 사람인식 됨, 예측모델 가동")
        
            print("\npreds:", preds, "\n")
        
            env_start_time = dt.datetime.now()
            env_start_time = env_start_time.strftime("%Y-%m-%d-%H-%M-%S")
            print("env_start_time:", env_start_time)
            
            
            # 사람 없으면
            for i in preds:
                if i == ['none']:
                    cnt += 1
        
            if cnt == 2:
                print("\n*******no person*******\n")
                system_next = 0
                occupant = 0
                not_occu += 1
        
                # 전 단계에서 시스템이 켜져있으면 off 시그널 보내기
                if system_now != 0 and not_occu >= 5:
                    control_start_time = 0
                    for i in range(2):
                        # <-----------------IR
                        port_IR = serial.Serial(ir_tcp, 9600)
                        port_IR.write([0])
                        time.sleep(2)
                    print("IR signal sended")
                    system_now = system_next  # 0
                    saving_time = 0
                print("사람없음 종료")
                
                ac_sql = 'INSERT INTO ac(time, ac_onoff) VALUES (%s, %s)'
                ac_val = env_start_time, system_next
        
                pmv_sql= 'INSERT INTO PMV(time, occ) VALUES (%s, %s)'
                pmv_val= env_start_time, occupant
        
                db_upload(ac_sql, ac_val)
                db_upload(pmv_sql, pmv_val)
        
            # 사람 있으면
            else:
                print("preds[0]:", preds[0], "preds[1]:", preds[1])
                print("\n*******person exist*******\n")
                not_occu = 0
        
                # on/off (1/0)
                system_next = 1 # on
                # occupant (1/0)
                occupant = 1
        
                if len(preds[0]) >= 2 and len(preds[1]) >= 2:
                    if len(preds[0]) >= 3:
                        gar11=preds[0][1]
                        gar12=preds[0][2]
                    else:
                        gar11=preds[0][1]
                        gar12=None
        
                    if len(preds[1]) >= 3:
                        gar21=preds[1][1]
                        gar22=preds[1][2]
        
                    else:
                        gar21=preds[1][1]
                        gar22=None
                    print("two cam predicted")
        
                elif len(preds[0]) >= 2:
                    if len(preds[0]) >= 3:
                        gar11=preds[0][1]
                        gar12=preds[0][2]
                    else:
                        gar11=preds[0][1]
                        gar12=None
                    gar21=None; gar22=None;
                    print("cam01 predicted")
        
                elif len(preds[1]) >= 2:
                    if len(preds[1]) >= 3:
                        gar21=preds[1][1]
                        gar22=preds[1][2]
                    else:
                        gar21=preds[1][1]
                        gar22=None
                    gar11=None; gar12=None;
                    print("cam02 predicted")
        
                else:
                    print("nothing predicted")
        
                # 처음 사람이 인지된 순간
                # 전 단계에서 시스템이 꺼져있으면 on 시그널 보내기 (이때, default: 25도, 바람 1단계)
                if system_now == 0:
        
                    # 사람이 처음 인지된 순간부터 제어 시작 시간대 설정
                    control_start_time = dt.datetime.now()
                    control_start_time_format = control_start_time.strftime("%Y-%m-%d %H:%M:%S")
        
                    if saving_time == 0:
                        saving_time = control_start_time
                    else:
                        pass
        
                    # IR 제어 전송
                    for i in range(2):
                        # <-----------------IR + Power
                        port_IR = serial.Serial(ir_tcp, 9600)
                        port_IR.write([next_setTA]) # initial value: 22
                        time.sleep(2)
                    print("IR signal sended")
                    
                    control_state='new_start'
                    system_now = system_next  # 1
                # 전 단계에서 시스템이 켜져있지만, 제어가 다시 시작될때 (40분 제어 term 이후 새로운 제어 시작 시)
                elif system_now != 0 and control_start_time == 0:
                    control_start_time = dt.datetime.now()
                    control_start_time_format = control_start_time.strftime("%Y-%m-%d %H:%M:%S")
                    control_state=None
                
                else: control_state=None
                
                ac_sql = 'INSERT INTO ac(time, ac_onoff, ac_sp, ac_mode, ac_finish) VALUES (%s, %s, %s, %s, %s)'
                ac_val = env_start_time, system_next, next_setTA, ac_mode, control_state
        
                pmv_sql= 'INSERT INTO PMV(time, occ, cam1_clo, cam2_clo) VALUES (%s, %s, %s, %s)'
                pmv_val= env_start_time, occupant, str([gar11, gar12]), str([gar21, gar22])
        
                db_upload(ac_sql, ac_val)
                db_upload(pmv_sql, pmv_val)
        
                print('time', 'Inserted')
        
            print("try once")
    
            
            # 제어 시작 시점이 초기값이 아니면서 제어 시간을 만족하면 BREAK
            if control_start_time != 0 and (dt.datetime.now()-control_start_time).seconds >= 60:
                control_start_time = 0
                break
        
            else:
                continue
        
    
            
                
        ################################################## 제어 시작###################################
            
        print("\n *********** control started ********* \n")
        
        # 아두이노 센서 & IR 작동 켜두기
        if system_now == 1:
            #DB 데이터 불러오기 
            ac_sql = "SELECT * FROM ac ORDER BY time DESC LIMIT 10"
            ac_col = ['time', 'ac_onoff', 'ac_sp', 'ac_mode', 'ac_finish']
            
            indoor_sql = "SELECT * FROM indoor ORDER BY time DESC LIMIT 10"
            indoor_col = ['time','in1_temp', 'in1_humid', 'in1_mrt', 'in1_air', 'in1_co2', 'in1_pm10', 'in1_pm2_5', 'in2_temp', 'in2_humid', 'in2_mrt', 'in2_air', 'in2_co2', 'in2_pm10', 'in2_pm2_5', 'in2_watt']
            
            pmv_sql = "SELECT * FROM PMV ORDER BY time DESC LIMIT 10"
            pmv_col = ['time', 'occ', 'cam1_clo', 'cam2_clo', 'cam1_met', 'cam2_met', 'clo', 'met', 'pmv']
            
            met_sql = "SELECT * FROM MET ORDER BY time DESC LIMIT 10"
            met_col = ['time', 'met1', 'met2']            
            
            result_ac=db_download(ac_sql, ac_col)
            result_indoor=db_download(indoor_sql, indoor_col)
            result_pmv=db_download(pmv_sql, pmv_col)
            result_met=db_download(met_sql, met_col)
            
            operate=list(result_ac['ac_onoff'])
            operate.reverse()
            occ_idx=operate.index(1)
            
            ac_time=list(result_ac['time'])
            ac_time.reverse()
            start_time=ac_time[occ_idx]
            
            # 재실 시간 환경변수 평균 값 산출 
            indoor_time=result_indoor['time']
            pmv_time=result_pmv['time']
            met_time=result_met['time']
            
            indoor_idx=[i for i in range(len(indoor_time)) if indoor_time[i] > start_time]
            pmv_idx=[i for i in range(len(pmv_time)) if pmv_time[i] > start_time]
            met_idx=[i for i in range(len(met_time)) if met_time[i] > start_time]
            
            indoor_data=result_indoor[indoor_idx[0]:indoor_idx[-1]+1]
            pmv_data=result_pmv[pmv_idx[0]:pmv_idx[-1]+1]
            met_data=result_met[met_idx[0]:met_idx[-1]+1]
            
            temp=indoor_data['in1_temp'].append(indoor_data['in2_temp'])
            humid=indoor_data['in1_humid'].append(indoor_data['in2_humid'])
            mrt=indoor_data['in1_mrt'].append(indoor_data['in2_mrt'])
            vel=indoor_data['in1_air'].append(indoor_data['in2_air'])
            
            # 환경데이터 outlier 제거
            temp=[i for i in temp if 10 < i < 50]
            humid=[i for i in humid if 0 < i]
            mrt=[i for i in mrt if 10 < i < 50]
            vel=[i for i in vel if 0 <= i]
            
            av_temp = round(sum(temp)/len(temp), 2)
            av_humid = round(sum(humid)/len(humid), 2)
            av_mrt = round(sum(mrt)/len(mrt), 2)
            av_vel = round(sum(vel)/len(vel), 2)
            
            # 대표 CLO 산출
            clo=pmv_data['cam1_clo'].append(pmv_data['cam2_clo'])
            clo=clo.tolist()
            CLO=[ast.literal_eval(i) for i in clo if not i is None]
            CLO=sum(CLO, [])
            CLO=[i for i in CLO if i != None]
            
            if not CLO: ref_clo = 1.0
            else: ref_clo = best_clo_ensemble(CLO)
            
            # 대표 MET 산출
            met=met_data['met1'].append(met_data['met2'])
            met=met.tolist()
            print("met_time:", met_data['time'])
            print("met 1, 2 to list:", met)
            MET=[ast.literal_eval(i) for i in met if not i is None]
            MET=sum(MET, [])
            MET=[i for i in MET if i != None]
            
            if not MET: ref_met = 1.0
            else: ref_met = Counter(MET).most_common()[0][0]
            
            print('temp: ', av_temp, '\thumid: ', av_humid, '\tmrt: ', av_mrt, '\tvel: ', av_vel)
            
            # 현재 PMV 계산
            now_pmv=comfort.models.pmv(av_temp, av_mrt, av_vel, av_humid, ref_met, ref_clo)
            print('CLO: ', ref_clo, '\tMET: ', ref_met, '\tnow PMV: ', now_pmv)
            
            #next set temp 설정
            next_pmv=[]
            temprange=[18,19,20,21,22,23,24,25,26,27,28,29,30]
            for TA in temprange:
                pmv_cal=comfort.models.pmv(TA, TA, av_vel, av_humid, MET, clo_ensemble)
                next_pmv.append(pmv_cal)
                
            next_pmv=np.asarray(next_pmv)
            idx=(abs(next_pmv)).argmin()
            best_pmv=next_pmv[idx]
            next_setTA=temprange[idx]
            
            #DB업로드
            pmv_update_sql='UPDATE PMV SET clo = %s, met = %s, pmv =%s WHERE time = %s'
            pmv_update_val=clo_ensemble, MET, now_pmv, env_start_time
            
            
            db_upload(pmv_update_sql, pmv_update_val)
            
            #신호 보내기
            if (dt.datetime.now()-saving_time).seconds >= 120:
            
                for i in range(2):
                    port_IR = serial.Serial(ir_tcp, 9600)  # <-----------------IR + Power
                    port_IR.write([0])
                    time.sleep(2)
                saving_time=0  # 제어 사이클 종료
                system_now=0   # 시스템 꺼짐
                
                ac_update_sql='UPDATE ac SET ac_sp = %s, ac_finish = %s WHERE time = %s'
                ac_update_val=str(0), 'finished', env_start_time
                db_upload(ac_update_sql, ac_update_val)
                print("IR OFF signal sended")
                print("Thie Ensemble Case Finished...")
                print("Next Ensemble Case Started!!") 
                break
            
            else: 
                # IR 제어 전송
                for i in range(2):
                    port_IR = serial.Serial(ir_tcp, 9600)  # <-----------------IR + Power
                    port_IR.write([next_setTA])
                    time.sleep(2)
                print("IR signal sended")
                
                ac_update_sql='UPDATE ac SET ac_sp = %s, ac_finish = %s WHERE time = %s'
                ac_update_val=next_setTA, 'new_signal', env_start_time
                db_upload(ac_update_sql, ac_update_val)
    time.sleep(180)
    not_occu=0