python cam.py -cam 268 -f./datal/dark_m10_60.0__gain1_ -gain 1 -exp 60.0 -auto 1 -count 20
python cam.py -cam 268 -f./datal/dark_m10_60.0__gain50_ -gain 50 -exp 60.0 -auto 1 -count 20
python cam.py -cam 268 -f./datal/dark_m10_60.0__gain56_ -gain 56 -exp 60.0 -auto 1 -count 20
python cam.py -cam 268 -f./datal/dark_m10_60.0__gain80_ -gain 80 -exp 60.0 -auto 1 -count 20
python cam.py -cam 268 -f./datal/dark_m10_60.0__gain90_ -gain 90 -exp 60.0 -auto 1 -count 20
python cam.py -cam 268 -f./datal/dark_m10_60.0__gain95_ -gain 95 -exp 60.0 -auto 1 -count 20
python cam.py -cam 268 -f./datal/dark_m10_60.0__gain100_ -gain 100 -exp 60.0 -auto 1 -count 20
python cam.py -cam 268 -f./datal/dark_m10_60.0__gain120_ -gain 120 -exp 60.0 -auto 1 -count 20


python mean.py ./datal/dark_m10_60.0__gain1_*.ser ./datal/dark_m10_60.0__gain1.fits
python mean.py ./datal/dark_m10_60.0__gain50_*.ser ./datal/dark_m10_60.0__gain50.fits
python mean.py ./datal/dark_m10_60.0__gain56_*.ser ./datal/dark_m10_60.0__gain56.fits
python mean.py ./datal/dark_m10_60.0__gain80_*.ser ./datal/dark_m10_60.0__gain80.fits
python mean.py ./datal/dark_m10_60.0__gain90_*.ser ./datal/dark_m10_60.0__gain90.fits
python mean.py ./datal/dark_m10_60.0__gain95_*.ser ./datal/dark_m10_60.0__gain95.fits
python mean.py ./datal/dark_m10_60.0__gain100_*.ser ./datal/dark_m10_60.0__gain100.fits
python mean.py ./datal/dark_m10_60.0__gain120_*.ser ./datal/dark_m10_60.0__gain120.fits

rm ./datal/*.ser

python cam.py -cam 268 -f./datal/dark_m10_30.0__gain1_ -gain 1 -exp 30.0 -auto 1 -count 20
python cam.py -cam 268 -f./datal/dark_m10_30.0__gain50_ -gain 50 -exp 30.0 -auto 1 -count 20
python cam.py -cam 268 -f./datal/dark_m10_30.0__gain56_ -gain 56 -exp 30.0 -auto 1 -count 20
python cam.py -cam 268 -f./datal/dark_m10_30.0__gain80_ -gain 80 -exp 30.0 -auto 1 -count 20
python cam.py -cam 268 -f./datal/dark_m10_30.0__gain90_ -gain 90 -exp 30.0 -auto 1 -count 20
python cam.py -cam 268 -f./datal/dark_m10_30.0__gain95_ -gain 95 -exp 30.0 -auto 1 -count 20
python cam.py -cam 268 -f./datal/dark_m10_30.0__gain100_ -gain 100 -exp 30.0 -auto 1 -count 20
python cam.py -cam 268 -f./datal/dark_m10_30.0__gain120_ -gain 120 -exp 30.0 -auto 1 -count 20


python mean.py ./datal/dark_m10_30.0__gain1_*.ser ./datal/dark_m10_30.0__gain1.fits
python mean.py ./datal/dark_m10_30.0__gain50_*.ser ./datal/dark_m10_30.0__gain50.fits
python mean.py ./datal/dark_m10_30.0__gain56_*.ser ./datal/dark_m10_30.0__gain56.fits
python mean.py ./datal/dark_m10_30.0__gain80_*.ser ./datal/dark_m10_30.0__gain80.fits
python mean.py ./datal/dark_m10_30.0__gain90_*.ser ./datal/dark_m10_30.0__gain90.fits
python mean.py ./datal/dark_m10_30.0__gain95_*.ser ./datal/dark_m10_30.0__gain95.fits
python mean.py ./datal/dark_m10_30.0__gain100_*.ser ./datal/dark_m10_30.0__gain100.fits
python mean.py ./datal/dark_m10_30.0__gain120_*.ser ./datal/dark_m10_30.0__gain120.fits



rm ./datal/*.ser

python cam.py -cam 268 -f./datal/bias_gain30_ -gain 30 -exp 0.000001 -auto 1 -count 300
python cam.py -cam 268 -f./datal/bias_gain56_ -gain 56 -exp 0.000001 -auto 1 -count 300
python cam.py -cam 268 -f./datal/bias_gain80_ -gain 80 -exp 0.000001 -auto 1 -count 300
python cam.py -cam 268 -f./datal/bias_gain105_ -gain 105 -exp 0.000001 -auto 1 -count 300


python mean.py ./datal/bias_gain30_*.ser ./datal/bias_gain30.fits
python mean.py ./datal/bias_gain56_*.ser ./datal/bias_gain56.fits
python mean.py ./datal/bias_gain80_*.ser ./datal/bias_gain80.fits
python mean.py ./datal/bias_gain105_*.ser ./datal/bias_gain105.fits

rm ./datal/*.ser
