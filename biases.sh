rm ./datal/bias*.ser

python cam.py  -f ./datal/bias_gain30_ -gain 30 -exp 0.000001 -auto 1 -count 300
python cam.py  -f ./datal/bias_gain80_ -gain 80 -exp 0.000001 -auto 1 -count 300
python cam.py  -f ./datal/bias_gain105_ -gain 105 -exp 0.000001 -auto 1 -count 300


python mean.py ./datal/bias_gain30_*.ser ./datal/bias_gain30.fits
python mean.py ./datal/bias_gain80_*.ser ./datal/bias_gain80.fits
python mean.py ./datal/bias_gain105_*.ser ./datal/bias_gain105.fits

rm ./datal/bias *.ser
