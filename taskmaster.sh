python live.py --alias="LIVE_RAND" --seed=12 --app="app_1" --pro="pro_1" --load_policy=0 --load_alias=''
python live.py --alias="LIVE_BIRD" --seed=12 --app="app_1" --pro="pro_1" --load_policy=1 --load_alias='META'

python metb.py
nohup python metb.py > metb.out &

nohup python live.py --alias="LIVE_RAND" --seed=12 --app="app_1" --pro="pro_1" --load_policy=0 --load_alias='' > rand_app1.out &
nohup python live.py --alias="LIVE_BIRD" --seed=12 --app="app_1" --pro="pro_1" --load_policy=1 --load_alias='META' > meta_app1.out &

nohup python live.py --alias="LIVE_RAND2" --seed=12 --app="app_2" --pro="pro_1" --load_policy=0 --load_alias='' > rand_app2.out &
nohup python live.py --alias="LIVE_BIRD2" --seed=12 --app="app_2" --pro="pro_1" --load_policy=1 --load_alias='META' > meta_app2.out &

nohup python live.py --alias="LIVE_RAND3" --seed=12 --app="app_3" --pro="pro_1" --load_policy=0 --load_alias='' > rand_app3.out &
nohup python live.py --alias="LIVE_BIRD3" --seed=12 --app="app_3" --pro="pro_1" --load_policy=1 --load_alias='META' > meta_app3.out &