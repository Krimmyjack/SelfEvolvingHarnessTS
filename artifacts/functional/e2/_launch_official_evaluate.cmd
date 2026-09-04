@echo off
setlocal
cd /d C:\Users\辉\Desktop\Agent\SelfEvolvingHarnessTS-deepseek-guidance-evolution
set PYTHONUNBUFFERED=1
D:\Anaconda_envs\envs\project\python.exe -u evaluation\functional\run_e2_t6_natural_a5_a3.py --evaluate
endlocal
