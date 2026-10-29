# Maglev with Camera
These codes are for using a machine vision camera to do magnetic levitation.
## Part of Hardware
Raspberry pi 4B or 5, HT-SUA33GM-T1V-C(https://www.huatengvision.com/product/195/), ADS1115, INA240 A2.
## Implenment Detail
1. Install the preemptRT for your Raspberry pi.(I used Linux raspberrypi 5.10.110-rt65-v7l+ #1 SMP PREEMPT_RT, PRETTY_NAME="Raspbian GNU/Linux 10 (buster)"
NAME="Raspbian GNU/Linux"
VERSION_ID="10"
VERSION="10 (buster)"
VERSION_CODENAME=buster
ID=raspbian
ID_LIKE=debian.)
2. Install the linux SDK for HT-SUA33GM-T1V-C.
3. Install the dependant libraries in the Requirements.txt, for Raspberry pi 5, it needs different setting in hardware PWM(https://github.com/Pioreactor/rpi_hardware_pwm/issues/10#issuecomment-2500952367).
## Codes Detail
### Camera Sensor
1. Camera sensor ROI test: Pendulum_GrabAPhoto.py
2. Camera senor time delay identification: PendulumTest_Software_timestatics.py
### Close Loop Control
1. Multi-steps case: Cam_Pos_FFPIDCOM_V_PI_HB_MS.py
2. Sine wave in fixed frequency case: Cam_Pos_FFPIDCOM_V_PI_HB_RS.py
3. Sine wave in changed frequency case: Cam_Pos_FFPIDCOM_V_PI_HB_RSC.py
