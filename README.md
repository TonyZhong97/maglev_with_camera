# Maglev with Camera
These codes are for using a machine vision camera to do magnetic levitation.
## Part of Hardware
Raspberry pi 4B or 5, HT-SUA33GM-T1V-C(https://www.huatengvision.com/product/195/), ADS1115.
## Implenment Detail
1. Install the preemptRT for your Raspberry pi.
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
