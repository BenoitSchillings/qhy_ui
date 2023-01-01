python cam.py  -f ./datal/dark_m5_0.1__gain1_ -gain 1 -exp 0.1 -auto 1 -count 300
python cam.py  -f ./datal/dark_m5_0.1__gain50_ -gain 50 -exp 0.1 -auto 1 -count 300
python cam.py  -f ./datal/dark_m5_0.1__gain100_ -gain 100 -exp 0.1 -auto 1 -count 300
python cam.py  -f ./datal/dark_m5_0.1__gain120_ -gain 120 -exp 0.1 -auto 1 -count 300
python cam.py  -f ./datal/dark_m5_0.1__gain150_ -gain 150 -exp 0.1 -auto 1 -count 300
python cam.py  -f ./datal/dark_m5_0.1__gain200_ -gain 200 -exp 0.1 -auto 1 -count 300
python cam.py  -f ./datal/dark_m5_0.1__gain250_ -gain 250 -exp 0.1 -auto 1 -count 300



python mean.py ./datal/dark_m5_0.1__gain1_*.ser ./datal/dark_m5_0.1__gain1.fits
python mean.py ./datal/dark_m5_0.1__gain50_*.ser ./datal/dark_m5_0.1__gain50.fits
python mean.py ./datal/dark_m5_0.1__gain120_*.ser ./datal/dark_m5_0.1__gain120.fits
python mean.py ./datal/dark_m5_0.1__gain150_*.ser ./datal/dark_m5_0.1__gain150.fits
python mean.py ./datal/dark_m5_0.1__gain200_*.ser ./datal/dark_m5_0.1__gain200.fits
python mean.py ./datal/dark_m5_0.1__gain250_*.ser ./datal/dark_m5_0.1__gain250.fits