# Deeplearning_Language-Modeling
인공신경망과 딥러닝 강의 과제용

**1. Plot the average loss values for training and validation.
- vanilla RNN Train & Validation Loss**
Epoch: 1, Train Loss: 1.8371
Validation Loss: 1749.9075
Epoch: 2, Train Loss: 1.8127
Validation Loss: 1808.2797
Epoch: 3, Train Loss: 1.9337
Validation Loss: 1951.4778
Epoch: 4, Train Loss: 2.0807
Validation Loss: 2048.5984
Epoch: 5, Train Loss: 2.2213
Validation Loss: 2141.1007
Epoch: 6, Train Loss: 2.2802
Validation Loss: 2272.1278
Epoch: 7, Train Loss: 2.3595
Validation Loss: 2277.3161
Epoch: 8, Train Loss: 2.4082
Validation Loss: 2312.0296
Epoch: 9, Train Loss: 2.4349
Validation Loss: 2361.8124
Epoch: 10, Train Loss: 2.4642
Validation Loss: 2376.5158

![Figure_2_1](https://github.com/NayunKim25/Deeplearning_Language-Modeling/assets/144984333/613030d3-017c-4b17-a9e8-bde813fb63da)
Loss 값을 살펴보면 vanilla RNN의 경우, 검증 데이터에 대한 손실 값이 지속적으로 증가하고 있으며 마지막 epoch에서는 2376.5158로 첫 epoch와의 차이가 약 627로 크게 변화하는 것을 확인할 수 있다.
이를 통해 vanilla RNN은 시퀀스 길이가 길어짐에 따라 장기 의존성을 잘 학습하지 못하고, overfitting이 발생하였음을 알 수 있다.

**- LSTM Train & Validation Loss**
Epoch: 1, Train Loss: 1.6060
Validation Loss: 1609.2995
Epoch: 2, Train Loss: 1.4546
Validation Loss: 1609.4558
Epoch: 3, Train Loss: 1.4290
Validation Loss: 1624.6096
Epoch: 4, Train Loss: 1.4198
Validation Loss: 1627.2432
Epoch: 5, Train Loss: 1.4162
Validation Loss: 1628.1502
Epoch: 6, Train Loss: 1.4162
Validation Loss: 1623.1195
Epoch: 7, Train Loss: 1.4194
Validation Loss: 1629.5082
Epoch: 8, Train Loss: 1.4238
Validation Loss: 1632.7086
Epoch: 9, Train Loss: 1.4275
Validation Loss: 1634.2278
Epoch: 10, Train Loss: 1.4381
Validation Loss: 1644.5962
Vanilla RNN Validation Loss: 2376.5158
LSTM Validation Loss: 1644.5962

![Figure_2_2](https://github.com/NayunKim25/Deeplearning_Language-Modeling/assets/144984333/2a046730-97dd-478d-b2d0-9ffe33e61bfb)
Loss 값을 살펴보면 LSTM의 경우, 검증 데이터에 대한 손실 값이 비교적 일정하게 유지되며 vanilla RNN에 비해 낮은 손실 값을 보이는 것을 확인할 수 있다.
이를 통해 LSTM이 긴 시퀀스의 의존성을 더 잘 처리하고, 더 안정적으로 학습되는 것을 알 수 있다.

**2. Try different temperatures when you generate characters, and discuss what difference the temperature makes and why it helps to generate more plausible results.**

Temperature 값을 0.5, 1, 1.5로 설정하여 출력 결과를 비교하였다.

1) Temperature: 0.5
The blood did the war, whom we shall he should be so blame the consul; when my soul all the true like a 
The king against the which the tribunes, for this are consul, and the people sent, and the comfort, sir,
The interious soul shall hear him that deserved some corn of the people that or stood me to see their ow
The consul. I have been the senators, and do not your country, and his state, when they say, my soul to 
The gods stood with my soul common serves the gods, the grace.LADY ANNE:I will receive my soul show the 
==========================================================================================================

2) Temperature: 1.0
The too old pleasions, sir: we content mark!Lest it to rule interrandon with you all by suippy!We are mo
The idle highness before thee!COMINIUS:Embsel, the gods she,I'll fight:Will beA sir, and as the crost fi
The bowels mosse you.Wheneport have been our unto the weaguins. If are an eyes and Marcius;ise, country,
The godnengs in his viore unhatch the quarrel is for him but his way;s if 'fore, to rock in makes?--It i
The offir'd soul,Tell with theeBut be his wife therefore us thatsaly,Yet: thou show'd and proper use,Mak
==========================================================================================================

3) Temperature: 1.5
The rightly this wise's leave you end i'erThat the enmenoneLikiussOf him he'ld battle fOverseme!Which I 
The boRe:But whenbake aid sGo; smean?First Servingmen.My 'lagun of devisuits andSonly?ow let by turn me?
The sword, that, toHath aul noe. . I charged thee among: our abut that? Only,Wine purse.LAMYers loving t
The gravePrinctUct I do,Below thatWheeld it's after KhanHeart your sword:To oldh raves.GLOUCESTER:So lea
The opeman annerIAs too?I'll doCeference?First Murderer palp what,ir,sThat small, stick, abusines. Deil,
==========================================================================================================

기본적으로 Temperature는 생성된 텍스트의 랜덤성 또는 불확실성 수준을 조절한다. 
따라서 낮은 Temperature에서는 생성된 텍스트가 더 결정적이고 보수적인 경향을 보이며, Temperature를 높이면 생성 과정에 더 많은 무작위성이 도입된다.
실제 출력 결과를 비교해보면 낮은 Temperature의 결과가 높은 Temperature의 결과보다 일관되고 구조화되어있으며, 문법 및 구문 규칙을 더 잘 따르는 것을 확인할 수 있다.
Temperature가 높아질수록 무작위성이 높아져 무의미한 단어나 구문이 많아지고, 말이 안되는 단어와 구문이 많이 등장하여 이해할 수 없는 결과를 나타내는 것을 알 수 있다.
하지만 높은 Temperature는 더 다양한 가능성을 고려할 수 있기 때문에 창의적인 결과물을 얻을 수 있는 장점이 있다.
