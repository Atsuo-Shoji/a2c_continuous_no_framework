# A2C（Advantage Actor Critic）<br>ActorとCriticで中間層を共有／分離の2パターンを構築し比較<br>（フレームワーク不使用）

<br>ActorとCriticで中間層を共有する構成と、ActorとCriticを分離する構成の2パターンで構築し、比較しています。<br>
環境としては、OpenAI GymのPendulum、LunarLanderContinuous、BipedalWalker、pybullet-gymのHopperPyBulletを使用しています。<br>

構築に際しては、フレームワークを使用せず、主にnumpyだけを使用しました。<br>

<br>

#### <<未訓練モデル/訓練済モデルでplayした結果の比較>>

<br>

| LunarLanderContinuous　未訓練モデルでPlay<br>枠外に飛去またはゴール外に落下 | LunarLanderContinuous　訓練済モデルでPlay<br>ゴール内に水平を保って着陸 |
|      :---:       |     :---:      |
|![LunarLanderc_beginner](https://user-images.githubusercontent.com/52105933/95009756-6882e280-065f-11eb-8d15-51e64f56b7b6.gif)|![LunarLanderc_step_202010022143_2_138s](https://user-images.githubusercontent.com/52105933/95009767-7c2e4900-065f-11eb-8182-a271390bf9df.gif)|

| BipedalWalker　未訓練モデルでPlay<br>すぐ転倒し前に進めない | BipedalWalker　訓練済モデルでPlay<br>拙いながら何とか前進 |
|      :---:       |     :---:      |
|![BipedalWalker_beginner_66s](https://user-images.githubusercontent.com/52105933/95009942-9c123c80-0660-11eb-9cb1-b5ee0a2a90f7.gif)|![BipedalWalker_202010011308](https://user-images.githubusercontent.com/52105933/95009953-b1876680-0660-11eb-9865-939f1559bc85.gif)|

| Pendulum　未訓練モデルでPlay<br>バーは一向に立たない | Pendulum　訓練済モデルでPlay<br>バーは途中から直立を維持 |
|      :---:       |     :---:      |
|![Pendulum_sep_biginner](https://user-images.githubusercontent.com/52105933/95010103-b3055e80-0661-11eb-9969-15166ff55da4.gif)|![Pendulum_sep_202010010221](https://user-images.githubusercontent.com/52105933/95010122-d6c8a480-0661-11eb-8703-64dcad0b2d35.gif)|

| HopperPyBullet　未訓練モデルでPlay<br>すぐ転倒し前に進めない | HopperPyBullet　訓練済モデルでPlay<br>上限1000ステップまで元気に前進 |
|      :---:       |     :---:      |
|![HopperPyBullet_test](https://user-images.githubusercontent.com/52105933/95031842-07a4ea00-06f3-11eb-9022-db54e399da32.gif)|![Hopper_std_G_202010122350_1000_1579](https://user-images.githubusercontent.com/52105933/95936264-218bae80-0e10-11eb-9804-48b9add13747.gif)|

※HopperPyBulletの画像が白いのはご了承下さい。Google Colaboratory上で3Dのpybullet-gymの動きを色つきでキャプチャーする簡便な方法がありませんでした。<BR>
![hopper_60](https://user-images.githubusercontent.com/52105933/95032126-6159e400-06f4-11eb-81b6-0762ce248401.png) ←本来のHopperPyBulletの外観。脳内イメージ補完してください。<BR><BR>

## 概要
ActorとCriticで中間層を共有する構成と、ActorとCriticを分離する構成の2パターンで構築し、比較しています。<br>
**強化学習とA2Cを材料にして、マルチタスク学習とシングルタスク学習の比較**をしているものだと思ってください。<BR>
    
本モデルは、行動が連続値を取る環境を対象としています。<BR>
本稿では、OpenAI GymのPendulum、LunarLanderContinuous、BipedalWalker、pybullet-gymのHopperPyBulletを使用しています。<br>
    
また、フレームワークを使用せず、主にnumpyだけでA2Cを構築しています。<br>

※本稿の「A2C」とは、純粋なAdvantage Actor Criticのことであり、分散処理は含みません<br><br>

### ActorとCriticで中間層を共有／分離の2パターンで比較
ActorとCriticで中間層を共有する構成と、ActorとCriticを分離する構成の2パターンで構築し、比較しています。
<br><br>
    
### フレームワークを使用せず実装
フレームワークを使用せずに主にnumpyだけで実装しています。<br>
Actorの損失、誤差逆伝播、その他諸々を自力で0から実装しています。 
<br><br>
    
### 対象とする環境
本モデルは、仕様上、以下の環境を対象としています。<BR>

| 状態 | 行動 |
|  :---:  |  :---:  |
|離散/連続|連続|

**基本的に上記を満たす環境なら動作します**が、環境毎の仕様や性質（状態や行動の次元数の大きさ、報酬設計など）により、訓練成果が変動します。<br>

本稿の実験においては、以下の環境を使用しています。<br>

| 環境名 | 外観| 状態の次元 | 行動の次元 |1エピソードでの上限ステップ数|目的|
|      :---:     |      :---:      |     :---:     |     :---:      |     :---:     |     :---:     |
|Pendulum|![pendulum_mini](https://user-images.githubusercontent.com/52105933/95575625-fa2c8e80-0a69-11eb-935c-38da16dc8afa.png)|3|1|上限は無し<br>固定で200ステップ/エピソード|バーの直立|
|LunarLanderContinuous|![LunarLanderContinuous_mini](https://user-images.githubusercontent.com/52105933/95575748-352ec200-0a6a-11eb-8494-997ac236f1ae.png)|8|2|1000|ゴール領域に着陸|
|HopperPyBullet|![hopper_mini](https://user-images.githubusercontent.com/52105933/95576384-50e69800-0a6b-11eb-9a18-711c04d690c9.png)|15|3|1000|歩いて（跳ねて？）遠くまで前進|
|BipedalWalker|![BipedalWalker_mini](https://user-images.githubusercontent.com/52105933/95576368-4cba7a80-0a6b-11eb-922e-52c584a8915e.png)|24|4|2000|歩いて遠くまで前進|

<br>
    
## ActorとCriticで中間層を共有／分離の2パターンの構成

### 両パターンの構成図
ActorとCriticで中間層を共有・・・「共有型」<br>
　マルチタスク学習：方策の最適化（Actor側）と価値関数の最適化（Critic側）の2つのタスクを単一NNで行う<br>
ActorとCriticを分離・・・「分離型」<br>
　シングルタスク学習：方策最適化のタスクは独立したActorのNNが、価値関数最適化のタスクは独立したCriticのNNが行う<br>
とし、以下のような構成とします。<br><br>

#### 構成のイメージ

<!--![NN概念図_60](https://user-images.githubusercontent.com/52105933/95010646-fc57ad00-0665-11eb-9642-78e02a63f2d8.png)-->
<!--![NN概念図_70](https://user-images.githubusercontent.com/52105933/95272801-516d0c00-087c-11eb-946f-baca485cb763.png)-->
![NN概念図_2_70](https://user-images.githubusercontent.com/52105933/95404720-c4e04d80-0950-11eb-885f-f6e88db5fa29.png)
<br>

#### 構成の概要図

<!--![NN概要図_共有型_60](https://user-images.githubusercontent.com/52105933/95010687-55274580-0666-11eb-9c4f-9c39c083c3ea.png)-->
![NN概要図_共有型_70](https://user-images.githubusercontent.com/52105933/95272897-95601100-087c-11eb-8e5a-c6da81bd1d7f.png)
<!--![NN概要図_共有型](https://user-images.githubusercontent.com/52105933/95273016-dc4e0680-087c-11eb-9b7d-d07fe49e59ab.png)-->
<br><br>
<!--![NN概要図_分離型_60](https://user-images.githubusercontent.com/52105933/95010700-6ff9ba00-0666-11eb-8bce-f7c5d492f7c0.png)-->
![NN概要図_分離型_70](https://user-images.githubusercontent.com/52105933/95273139-3a7ae980-087d-11eb-8991-764b9b97ca65.png)

<br><br>

### 前提

- Actor側のメソッドは方策勾配法（Policy Gradient）<br>
ガウス方策を採用<BR><BR>
- Actorの損失関数にAdvantageを使用<br>
Q関数（行動価値関数）を割引報酬和で近似（REINFORCEアルゴリズム）<BR><BR>
- Critic側の損失関数は平均2乗和誤差<br>
教師信号は割引報酬和とする<BR><BR>
- 割引報酬和は標準化して使用<br>
ただし、平均を0にしない<BR>
G' =  G / σ　（σは標準偏差）<BR><BR>
- action（行動）はK次元ベクトルであるが、それら各次元は互いに独立であるとする
<BR><BR>
- 共有型も分離型も、中間層の各レイヤーのノード数は、状態の次元数、行動の次元数に応じて決まる<br>
これら次元数が大きければノードも多い<BR><BR>
- NNのパラメーター更新はエピソード毎　＝　1イテレーション/エピソード
<BR><BR>
- （共有型のみ）全体の損失 = Actorの損失 + 重み係数 × Criticの損失　とする<br>
本稿の全実験において、この重み係数を1.0に統一する（比較のため）
<br>

#### Actorの損失関数

action（行動）をK次元ベクトルとし、各次元が互いに独立であると仮定すると、ガウス方策のもとでのActorの損失関数は以下となります。<br><br>
A：行動変数　　　S：状態変数　　　N：バッチデータ（1エピソード時系列データ）のサイズ <br>
k：actionのK次元ベクトルのk次元目　　　i：バッチデータ（1エピソード時系列データ）のi番目 <br>
μ：正規分布の平均　　　var：正規分布の分散<br>
advantage：割引報酬和Gi - 価値関数Vi

![Actor損失関数_50](https://user-images.githubusercontent.com/52105933/95014983-7cd9d600-0685-11eb-9459-a7fd1d3c9876.png)

<br>
Actorの損失関数の計算グラフ：<br>
（手書きですみません・・）<br>
図中の（advの逆伝播の矢印の上の）「x」印は、（共有型において）この逆伝播はしない、という意味です。<br>

![loss_actor_計算グラフ2_加工_75](https://user-images.githubusercontent.com/52105933/96193596-6be16c80-0f83-11eb-8d31-5ac68ca65b76.png) <br>

da = -dLa * advantage / N　として<br>
dμk = 2da * (Aik - μk) / vark<br>
dlog(vark) = da * ( (Aik - μk)^2 / vark - 1 )<br>

<b>※共有型において、なぜActorの損失をAdvantageを介してCritic側に逆伝播させないのか</b><br>
actionの推測は、K次元正規分布N(μ, var)からのサンプリングで行われます。<br>
その際、NNの出力値として使用するのはμとvarだけです。価値関数Vは使用しません。<br>
よって、Actorの損失関数の最小化にVを関与させる＝Critic側のV出力ノードを関与させると、（Vを使用しない）actionの推論時における最適なμとvarにはならないです。

<BR>
    
#### 割引報酬和の標準化

本稿の全実験において、訓練時にモデル内部で割引報酬和の標準化をしています。<BR>
（訓練関数train()の引数で標準化をするかしないかを指定できます）<br>

割引報酬和は、Advantageを介してCriticのlossになります。<br>
使用した4環境（endulum、LunarLanderContinuous、BipedalWalker、HopperPyBullet）いずれも、割引報酬和を標準化しないと、CriticのlossがActorのlossから大きく乖離してしまいます。<br>
特に1エピソードのステップ数が200ステップ固定のPendulumは、訓練開始時のCriticのlossは10数万にもなります（Actorのlossはマイナス数百程度）。<br>
このような状況にならないように、割引報酬和の標準化をしています。<br>
※試しに割引報酬和の標準化をせずに訓練してみましたが、共有型は全滅し、分離型もあまり機能しませんでした。<br>

ただし、本モデルが内部で行う割引報酬和の標準化では、**平均を0にしません。**<BR>
G' =  G / σ　（σは標準偏差）<BR>
平均を0にすると、せっかくプラスなのにマイナスに”ずらされて”しまう割引報酬和が出てきます。<BR>
マイナスの報酬は、「そのような行動を取るな」とモデルに教え込むことにつながります。<BR>
従って、割引報酬和の標準化では平均を0にしない仕様にしています。

<br><br>

## ActorとCriticで中間層を共有／分離の2パターンの比較

Pendulum、LunarLanderContinuous、BipedalWalker、HopperPyBullet　の4環境を、<br>
共有型、分離型それぞれで訓練しました。<br>
「score（スコア）」とは、1エピソードで得られた報酬の単純合計のことです。<br>
NNのパラメーター更新はエピソード毎　＝　1イテレーション/エピソード　です。<BR>
下記グラフの横軸は全てエピソードです。
<br><br>

### Pendulum

| 状態の次元 | 行動の次元 |1エピソードでの上限ステップ数|目的|
|     :---:      |     :---:      |     :---:      |     :---:      |
|3|1|上限は無く固定で200ステップ/エピソード|バーの直立|

#### 訓練記録の比較
<br>

![共有型と分離型の比較_Pendulum_70](https://user-images.githubusercontent.com/52105933/95834606-256bf200-0d78-11eb-9a08-14bd156f0d72.png)

| 共有型/分離型 | 稼得score合計 |稼得score最大値|
|     :---:      |     :---:      |     :---:      |
|共有型|-5310万|0|
|分離型|-7366万|0|

#### まとめ・考察など

共有型は分離型に比べてscoreの稼得量が多く、score最大値（0）付近に到達するスピードも早かったです。<br>
共有型・分離型とも、バーの直立を維持するのに十分なまでの訓練をすることができました。<br>

**共有型の方が分離型より優れた機能を発揮しており、マルチタスク学習がうまくいった**ように見えます。

<br>

### LunarLanderContinuous

| 状態の次元 | 行動の次元 |1エピソードでの上限ステップ数|目的|
|     :---:      |     :---:      |     :---:      |     :---:      |
|8|2|1000|ゴール領域に着陸|

#### 訓練記録の比較
<br>

![共有型と分離型の比較_LunarLanderConti_70](https://user-images.githubusercontent.com/52105933/95837612-bd1f0f80-0d7b-11eb-80ce-1906cebcc5a7.png)

| 共有型/分離型 | 稼得score合計 |稼得score最大値|
|     :---:      |     :---:      |     :---:      |
|共有型|80万|289|
|分離型|94万|282|

#### まとめ・考察など

稼得score合計については、分離型が共有型を上回りました。<br>
その差14万は平均すると14／エピソードであり、また14万程度のブレは試行する度に生じるので、この程度の差は誤差として、**共有型と分離型は同等の性能**、と見ます。<br>
共有型・分離型とも、ある程度のscoreに達すると、それ以上は伸びませんでした。<br>

共有型・分離型とも、ある程度のscoreに達すると頭打ちになってしまうのは、<br>
このLunarLanderContinuousという環境の性質によるものなのか（実はそれ以上のscoreを出すのは著しく難易度が高い）、<br>
モデル側の何らかの原因によるものなのか、<br>
不明です。<br>
「モデル側の何らかの原因によるもの」の場合、他の環境にも共通している現象ですので、後でまとめて記述します。

<br>

### HopperPyBullet

| 状態の次元 | 行動の次元 |1エピソードでの上限ステップ数|目的|
|     :---:      |     :---:      |     :---:      |     :---:      |
|15|3|1000|歩いて（跳ねて？）遠くまで前進|

#### 訓練記録の比較
<br>

![共有型と分離型の比較_HopperPyBullet_70](https://user-images.githubusercontent.com/52105933/95869264-27e54080-0da6-11eb-96aa-6ad444cb867d.png)

| 共有型/分離型 | 稼得score合計 |稼得score最大値|
|     :---:      |     :---:      |     :---:      |
|共有型|3619万|1689|
|分離型|2263万|1276|

#### まとめ・考察など

共有型は分離型に比べてscoreの稼得量が多く、scoreを伸ばすスピードも早かったです。<br>
分離型も、共有型より劣っているとは言え、scoreをしっかり稼得しています。

**共有型の方が分離型より優れた機能を発揮しており、マルチタスク学習がうまくいった**ように見えます。<br>

ただし、共有型・分離型とも、scoreがあるところから頭打ちになります。<br>
稼得scoreの最大値は、共有型＞分離型、と開きがあるので、<br>
共有型での頭打ちになる現象の原因は、LunarLanderContinuousと同様、「環境の性質（実はそれ以上のscoreを出すのは著しく難易度が高い）」or「モデル側」の双方が考えられますが、<br>
分離型での頭打ちになる現象の原因は、「モデル側」のみとなるでしょう。<br>
モデル側の原因については、他の環境にも共通している現象ですので、後でまとめて記述します。

<br>

### BipedalWalker

| 状態の次元 | 行動の次元 |1エピソードでの上限ステップ数|目的|
|     :---:      |     :---:      |     :---:      |     :---:      |
|24|4|2000|歩いて遠くまで前進|

#### 訓練記録の比較
<br>

![共有型と分離型の比較_BipedalWalker_70](https://user-images.githubusercontent.com/52105933/95916907-57b33900-0de4-11eb-90c3-be596506dcc6.png)

| 共有型/分離型 | 稼得score合計 |稼得score最大値|
|     :---:      |     :---:      |     :---:      |
|共有型|-350万|-60|
|分離型|-319万|-98|

#### まとめ・考察など

共有型と分離型の違いよりも、そもそも訓練されたのかどうかがよく分からない結果でした。<br>
イテレーションを2.5万回繰り返しても、稼得scoreの最大値にほぼ変化は無かったです。<br>

原因としては、環境BipedalWalkerの性質に訓練の仕方が合っていないから、かもしれません。<br>
BipedalWalkerの1エピソードの上限ステップ数は2000で、他の環境の2倍あります。<br>
また、BipedalWalkerの性質として、ステップ数は訓練の習熟度合いにあまり関係ない、というのがあります。<br>
訓練開始1エピソード目から最高2000ステップをいきなり叩き出したりします。<br>
つまり、訓練最中に、2000ステップという長大なエピソードが”まんべんなく”発生します。<br>
状態24次元＆行動4次元という高次元で、且つ2000ステップという長大なエピソードだと、エピソード毎の状態と行動のバリエーションがありすぎて、どのような（その長い一連の）行動を取ればそのような報酬がもたらされるのか、2.5万回程度のイテレーションではイマイチ学習し切れないのかもしれません。

<br>

### 全体のまとめ・考察など

大まかには、以下の現象となりました。<br>
「稼得scoreの絶対量」と「稼得scoreが伸びるスピード」において・・・
- **共有型は概ね分離型より良く機能した。**<br>
- 分離型も悪かったわけではなく、そこそこ機能した。<br>
- 共有型も分離型も、ある程度のところでscoreが伸びなくなった<br>

「共有型も分離型も、ある程度のところでscoreが伸びなくなった」のは、以下のような原因であると考えています。<br>
書いている順番に意味はありません。<br>

①実装のための仮定や近似<BR>

実装するために、仕方なく「そう仮定した」「そう近似した」ことがあります。<BR>

- action（行動）はK次元ベクトルであるが、それら各次元は互いに独立であるとする<br>
HopperPyBulletを例に取ります。<br>
行動は3次元ベクトルです。「同じ1本足」にある3つの関節に及ぼすトルクです。<br>
しかし、「同じ1本足」にある3つの関節は、独立の関係にあるはずはなく、互いに関係があるはずです。<br>
ではそのようにすればいいじゃないか、となりますが、数式展開が複雑になるので（そしてそれを実装しなくてはならないので）、現実的ではありません。<br>

- REINFORCEアルゴリズムによる近似やその他の近似<BR>
REINFORCEアルゴリズムにより、行動価値関数を割引報酬和で近似しています。<BR>
また、モンテカルロ近似を使用しています。<BR>
    
②共有型・分離型ともに表現力不足<br>

共有型・分離型ともに、状態の次元数と行動の次元数が増えるとNN中間層のノードも増えるような作りにしています。<br>
が、それではまだノードの増やし方が不十分なのかもしれません。<br>

ノードの増やし方ではなく、慢性的に表現力が不足していることも考えられます。<br>
以下のようにNNを拡張する、ということも一案です。<BR>

![NN概念図_新案_60](https://user-images.githubusercontent.com/52105933/96056598-fe213c00-0ec1-11eb-990f-6b90ba7b9d5d.png)

③共有型のCriticのlossの重み係数が不適切<br>

モデル全体のloss = Actorのloss + 重み係数 * Criticのloss <br>
本稿の全実験において、この重み係数をあえて1.0に統一しています。異なる環境間で比較するためです。<br>
が、本来はそれぞれの環境での適切な重み係数というのがあると思います。<br>

<BR>

## ディレクトリ構成・動かすのに必要な物
Planner_share.py<BR>
Planner_separate.py<BR>
common/<br>
&nbsp;└funcs.py<br>
&nbsp;└layers.py<br>
&nbsp;└optimizers.py<br>
trained_params/<br>
&nbsp;└（訓練済パラメーターのpickleファイル）<br>
-----------------------------------------------------------------------------------------------------<br>
- Planner_share.py：共有型モデル本体。中身はclass Planner_share です。モデルを動かすにはcommonフォルダが必要です。
- Planner_separate.py：分離型モデル本体。中身はclass Planner_separate です。モデルを動かすにはcommonフォルダが必要です。
<br>

## モデルの各ファイルの構成
Planner_share.pyのclass Planner_share、Planner_separate.pyのclass Planner_separateが、モデルの実体です。
<br><br>

![モデルの構成_70](https://user-images.githubusercontent.com/52105933/95835589-60225a00-0d79-11eb-8dd9-87f8cd0fe3c1.png)

このclass Planner_xx をアプリケーション内でインスタンス化して、環境の訓練やPlayといったpublicインターフェースを呼び出す、という使い方をします。
```
#モデルのインポート 
from Planner_share import * #モデル本体　共有型の場合
from Planner_separate import * #モデル本体　分離型の場合

#Pendulumの環境の活性化　ここではPendulumを使用
env = gym.make("Pendulum-v0")
  
#モデルのインスタンスを生成 
p_model_instance = Planner_share(name="hoge", env=env, state_dim=3, action_dim=1, wc_critic_loss=1.0) #共有型の場合
p_model_instance = Planner_separate(name="hoge", env=env, state_dim=3, action_dim=1) #分離型の場合

#以下、モデルインスタンスに対してそのpublicインターフェースを呼ぶ

#このモデルインスタンスの訓練 
result = p_model_instance.train(episodes=10000, steps_per_episode=200, gamma=0.99, metrics=1, softplus_to_advantage=False, weight_decay_lmd=0, verbose_interval=10)

#この訓練済モデルインスタンスにPendulumをPlayさせる 
try:

    curr_st = env.reset()
    env.render(mode='human')
    
    for st in range(200):
            
        #モデルインスタンスが最適な行動を推測
        action_predicted = p_model_instance.predict_best_action(curr_st) 
        
        #その行動を環境に指示
        next_st, reward, done, _ = env.step(action_predicted)

        #レンダリング（注意！Google ColaboratoryのようなGUI描画ウィンドウが無い実行環境では、このままでは実行できません。）
        env.render(mode='human')

        if done==True:
           #エピソード終了
           break

        curr_st = next_st

finally:
    env.close()

#この訓練済モデルインスタンスの訓練済パラメーターの保存
p_model_instance.save_params_in_file(file_dir=hoge, file_name=hoge)

#別の訓練済パラメーターをこのモデルインスタンスに読み込む
p_model_instance.overwrite_params_from_file(file_path=hoge)
```
<br><br>

## class Planner_share、class Planner_separate　のpublicインターフェース

#### class Planner_share、class Planner_separate　のpublicインターフェース一覧
| 名前 | 関数/メソッド/プロパティ | 機能概要・使い方 |
| :---         |     :---:      | :---         |
|Planner_share|     -      |（共有型）class Planner_share　のモデルインスタンスを生成する。<br>*model_instance* = Planner_share(name="hoge", env=env, state_dim=3, action_dim=1, wc_critic_loss=1.0)|
|Planner_separate|     -      |（分離型）class Planner_separate　のモデルインスタンスを生成する。<br>*model_instance* = Planner_separate(name="hoge", env=env, state_dim=3, action_dim=1)|
|train|     関数      |モデルインスタンスを訓練する。<br>result = *model_instance*.train(episodes=10000, steps_per_episode=200, gamma=0.99, metrics=0, softplus_to_advantage=False, weight_decay_lmd=0, verbose_interval=10)|
|predict_best_action|     関数      |モデルインスタンスが最適な行動を推測する。<br>best_action = *model_instance*.predict_best_action(a_state=hoge)|
|save_params_in_file|     関数      |モデルインスタンスのパラメーターをファイル保存する。<br>file_name = *model_instance*.save_params_in_file(file_dir=hoge, file_name=hoge)|
|overwrite_params_from_file|     メソッド      |モデルインスタンスのパラメーターを、ファイル保存された別のパラメーターで上書きする。<br>*model_instance*.overwrite_params_in_file(file_path=hoge)|
|env|     getterプロパティ      |モデルインスタンスが対象としている環境。インスタンス化時に指定された物。<br>env = *model_instance*.env|
|state_dim|     getterプロパティ      |モデルインスタンスが認識している、状態の要素数。インスタンス化時に指定された物。<br>state_dim = *model_instance*.state_dim|
|action_dim|     getterプロパティ      |モデルインスタンスが認識している、行動の要素数。インスタンス化時に指定された物。<br>action_dim = *model_instance*.action_dim|
|name|     getter/setterプロパティ      |モデルインスタンスの名前。<br>getter : hoge = *model_instance*.name<br>setter : *model_instance*.name = hoge|
|wc_critic_loss|     getter/setterプロパティ      |（共有型）モデルインスタンスのcritic lossの重み係数。<br>getter：hoge = *model_instance*.wc_critic_loss<br>setter : *model_instance*.wc_critic_loss = hoge|
<br>

### class Planner_share　のインスタンス化　（共有型）　*model_instance* = Planner_share(name, env, state_dim, action_dim, wc_critic_loss=1.0)
共有型であるclass Planner_shareのインスタンスを生成する。<br>
#### 引数：
| 名前 | 型 | 必須/既定値 | 意味 |
| :---         |     :---:      |     :---:     | :---         |
|name|文字列|必須|このモデルインスタンスの名前。|
|env|gym.wrappers.time_limit.TimeLimit|必須|環境のオブジェクトインスタンス。<br>例えばenv = gym.make("Pendulum-v0")　などと生成する。|
|state_dim|整数|必須|状態の次元数。<br>今後、このインスタンスは、状態の次元数はここで指定されたものであるという前提で挙動する。変更方法は無い。|
|action_dim|整数|必須|行動の次元数。<br>今後、このインスタンスは、行動の次元数はここで指定されたものであるという前提で挙動する。変更方法は無い。|
|wc_critic_loss|浮動小数点数|1.0|critic lossの重み係数。<br>モデル全体のloss = actor loss + wc_critic_loss * critic loss|

<br>

### class Planner_separate　のインスタンス化　（分離型）　*model_instance* = Planner_separate(name, env, state_dim, action_dim)
分離型であるclass Planner_separateのインスタンスを生成する。<br>
#### 引数：
| 名前 | 型 | 必須/既定値 | 意味 |
| :---         |     :---:      |     :---:     | :---         |
|name|文字列|必須|このモデルインスタンスの名前。|
|env|gym.wrappers.time_limit.TimeLimit|必須|環境のオブジェクトインスタンス。<br>例えばenv = gym.make("Pendulum-v0")　などと生成する。|
|state_dim|整数|必須|状態の次元数。<br>今後、このインスタンスは、状態の次元数はここで指定されたものであるという前提で挙動する。変更方法は無い。|
|action_dim|整数|必須|行動の次元数。<br>今後、このインスタンスは、行動の次元数はここで指定されたものであるという前提で挙動する。変更方法は無い。|

<br>

### ＜関数＞result = *model_instance*.train(episodes, steps_per_episode, gamma=0.99, metrics=1, softplus_to_advantage=False, weight_decay_lmd=0, verbose_interval=100)
モデルインスタンスを訓練します。<br>

#### 引数：
| 名前 | 型 | 必須/既定値 | 意味 |
| :---         |     :---:      |     :---:     | :---         |
|episodes|整数|必須|訓練に費やすエピソード数。|
|steps_per_episode|整数|必須|1エピソードで何ステップまで行うか。|
|gamma|浮動小数点数|0.99|報酬の現在価値算出のための割引率。|
|metrics|整数|1|このモデルの性能指標を指定する。0か1を指定する。<br>0：1エピソードあたりのステップ数（が多いほど性能が良い、と見なす）<br>1：1エピソードあたりのscore（が大きいほど性能が良い、と見なす）|
|standardize_G|boolean|True|割引報酬和を標準化するか。<br>※この標準化では平均を0にしない。σを標準偏差として、G'=G/σ という演算を行う。|
|softplus_to_advantage|boolean|False|Advantageにsoftplusを適用するか。<br>Advantageがマイナスになるのを防止したい場合に使用する。<br>※共有型の場合、actor lossからcritic側への逆伝播は行わないので、softplusの逆伝播も考慮する必要は無い。|
|weight_decay_lmd|浮動小数点数|0|荷重減衰のλ。|
|verbose_interval|整数|100|何エピソード毎にエピソード記録を出力するか。<br>0以下を指定すると一切出力しない。|

#### 戻り値「result」（Dictionary）の内部要素：
| key文字列 | 型 | 意味 |
| :---         |     :---:      | :---         |
|name|文字列|このモデルインスタンス名。|
|episode_count|整数|実際のエピソード数。|
|step_count_episodes|list|各エピソードのステップ数のエピソード毎の履歴。listの1要素は1エピソード。|
|loss_episodes|list|（共有型）各エピソードのモデル全体のlossのエピソード毎の履歴。listの1要素は1エピソード。|
|loss_actor_episodes|list|各エピソードのactor lossのエピソード毎の履歴。listの1要素は1エピソード。|
|loss_critic_episodes|list|各エピソードのcrtic lossのエピソード毎の履歴。listの1要素は1エピソード。|
|score_episodes|list|各エピソードのscore（稼得報酬の単純合計）のエピソード毎の履歴。listの1要素は1エピソード。|
|step_count_total|整数|実際のステップ総数。|
|processing_time_total|datetime.timedelta|総処理時間。|
|processing_time_total_string|文字列|総処理時間の文字列表現。|
|wc_critic_loss|浮動小数点数|（共有型）このモデルインスタンスに設定されたcritic lossの重み係数wc_critic_loss。|
|train()の引数|-|引数の指定値。|

<br>

#### 訓練中のベストなパラメーターを採用：<br>
1回の訓練中、エピソード1回毎にNNのパラメーターが更新されます。<BR>
たとえ訓練途中で”良い成績”を記録したエピソードがあったとしても、訓練終了時の最終エピソードの”成績が悪かった”場合、モデルインスタンスは”悪い”性能で訓練を終えた、ということになってしまいます。<BR>
そうならないために、ベストな成績を記録した直近のエピソード時のNNのパラメーターを一時退避しておき、訓練終了時にそのパラメーターを採用するようにしています。<BR>
”良い成績”、”悪い成績”とは、train()の引数の「metrics」（モデルの性能指標）で決まります。<br>
0：モデルの性能指標をステップ数とする⇒最多ステップ数を記録したエピソード時のパラメーターを最終的に採用<br>
1：モデルの性能指標をscoreとする⇒最大scoreを記録したエピソード時のパラメーターを最終的に採用
<BR><BR>
metrics=1（モデルの性能指標をscoreとした場合）
![bestパラメーター採用説明](https://user-images.githubusercontent.com/52105933/95319152-7ee4a480-08d2-11eb-89e4-9d305ba89926.png)

<br>

### ＜関数＞best_action = *model_instance*.predict_best_action(a_state)
与えられた状態（1個）での最適な行動を推測します。<br>

#### 引数：
| 名前 | 型 | 必須/既定値 | 意味 |
| :---         |     :---:      |     :---:     | :---         |
|a_state|ndarray<br>shapeは(インスタンス化時指定のstate_dim,)<br>又は(1, インスタンス化時指定のstate_dim)|必須|行動を取ろうとしている状態。1個のみ。<br>```a_state = env.reset()```<br>又は<br>```a_state, reward, done, _ = env.step(直前のaction)```<br>の戻り値を利用する。|

#### 戻り値：<BR>
- best_action<BR>
推測結果の行動。shapeは(インスタンス化時指定のaction_dim,)。<BR>
ここで得られたbest_actionは、以下のように環境に指示して、利用する。<br>
```next_state, reward, done, _ = env.step(best_action)```

<br>

### ＜関数＞file_name = *model_instance*.save_params_in_file(file_dir, file_name="")
現在のモデルインスタンスの訓練対象パラメーターとwc_critic_loss（共有型のみ）をpickleファイルに保存し、後に再利用できるようにします。<br>

#### 引数：
| 名前 | 型 | 必須/既定値 | 意味 |
| :---         |     :---:      |     :---:     | :---         |
|file_dir|文字列|必須|保存ファイルを置くディレクトリ。|
|file_name|文字列|空文字列|保存ファイル名（拡張子も含める）。<br>空文字列の場合、モデルインスタンス名.pickle　というファイル名になる。|

#### 戻り値：<BR>
- file_name<BR>
実際の保存ファイル名。

<br>

### ＜メソッド＞*model_instance*.overwrite_params_from_file(file_path)
pickleファイルに保存された訓練対象パラメーターとwc_critic_loss（共有型のみ）を読み込み、現在のモデルインスタンスのパラメーターを上書きします。<br>

#### 引数：
| 名前 | 型 | 必須/既定値 | 意味 |
| :---         |     :---:      |     :---:     | :---         |
|file_path|文字列|必須|保存ファイルのパス（拡張子も含める）。|

<br>

### ＜getterプロパティ＞*model_instance*.env
このモデルインスタンスが対象としている環境を返します。<br>
インスタンス化時に指定された物です。

<br>

### ＜getterプロパティ＞*model_instance*.state_dim
このモデルインスタンスが認識している、状態の次元数を返します。<br>
インスタンス化時に指定された物です。

<br>

### ＜getterプロパティ＞*model_instance*.action_dim
このモデルインスタンスが認識している、行動の次元数を返します。<br>
インスタンス化時に指定された物です。

<br>

### ＜getter/setterプロパティ＞*model_instance*.name
getterは、このモデルインスタンスの名前を返します。<br>
setterは、このモデルインスタンスの名前を設定します。<br>

#### setterが受け取る値：
| 型 |  意味 |
|     :---:      | :---         |
|文字列|モデルインスタンスの新しい名前。|

<br>

### ＜getter/setterプロパティ＞（共有型）*model_instance*.wc_critic_loss
wc_critic_lossとは、モデル全体のlossを算出する際の、criticに適用する重み係数のこと。<br>
モデル全体のloss = actor loss + wc_critic_loss * critic loss<br>
getterは、このモデルインスタンスのwc_critic_lossを返します。<br>
setterは、このモデルインスタンスのwc_critic_lossを設定します。<br>

#### setterが受け取る値：
| 型 |  意味 |
|     :---:      | :---         |
|浮動小数点数|モデルインスタンスの新しいwc_critic_loss。|

<br><br>

※本リポジトリに公開しているプログラムやデータ、リンク先の情報の利用によって生じたいかなる損害の責任も負いません。これらの利用は、利用者の責任において行ってください。