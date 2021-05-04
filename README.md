# A2C（Advantage Actor Critic）<br>ActorとCriticで中間層を共有／分離の2パターンを構築し比較<br>（フレームワーク不使用）

<br>ActorとCriticで中間層を共有する構成と、ActorとCriticを分離する構成の2パターンで構築し、訓練成果を比較しています。<br>
環境としては、OpenAI GymのPendulum、LunarLanderContinuous、BipedalWalker、pybullet-gymのHopperPyBulletを使用しています。<br>

構築に際しては、フレームワークを使用せず、主にnumpyだけを使用しました。<br>

<br>

#### 【未訓練モデル/訓練済モデルでplayした結果の比較】

<br>

| LunarLanderContinuous　未訓練モデルでPlay<br>枠外に飛去またはゴール外に落下 ||LunarLanderContinuous　A2C訓練済モデルでPlay<br>ゴール内に水平を保って着陸 |
|      :---:       |     :---:      |     :---:      |
|![LunarLanderc_beginner](https://user-images.githubusercontent.com/52105933/95009756-6882e280-065f-11eb-8d15-51e64f56b7b6.gif)|![矢印（赤）](https://user-images.githubusercontent.com/52105933/110228721-b0779f80-7f46-11eb-8cd9-469501beea50.png)|![LunarLanderc_step_202010022143_2_138s](https://user-images.githubusercontent.com/52105933/95009767-7c2e4900-065f-11eb-8182-a271390bf9df.gif)|

| BipedalWalker　未訓練モデルでPlay<br>すぐ転倒し前に進めない || BipedalWalker　A2C訓練済モデルでPlay<br>少しだけ何とか前進 |
|      :---:       |     :---:      |     :---:      |
|![BipedalWalker_beginner_66s](https://user-images.githubusercontent.com/52105933/95009942-9c123c80-0660-11eb-9cb1-b5ee0a2a90f7.gif)|![矢印（赤）](https://user-images.githubusercontent.com/52105933/110228721-b0779f80-7f46-11eb-8cd9-469501beea50.png)|![BipedalWalker_202103071440_436s_30r](https://user-images.githubusercontent.com/52105933/110274830-20ebf280-8013-11eb-9feb-94b3c9a4dc7a.gif)|

| Pendulum　未訓練モデルでPlay<br>バーは一向に立たない || Pendulum　A2C訓練済モデルでPlay<br>バーは途中から直立を維持 |
|      :---:       |     :---:      |     :---:      |
|![Pendulum_sep_biginner](https://user-images.githubusercontent.com/52105933/95010103-b3055e80-0661-11eb-9969-15166ff55da4.gif)|![矢印（赤）](https://user-images.githubusercontent.com/52105933/110228721-b0779f80-7f46-11eb-8cd9-469501beea50.png)|![Pendulum_sep_202010010221](https://user-images.githubusercontent.com/52105933/95010122-d6c8a480-0661-11eb-8703-64dcad0b2d35.gif)|

| HopperPyBullet　未訓練モデルでPlay<br>すぐ転倒し前に進めない || HopperPyBullet　A2C訓練済モデルでPlay<br>完璧な歩様で上限1000ステップまで前進 |
|      :---:       |     :---:      |     :---:      |
|![HopperPyBullet_test](https://user-images.githubusercontent.com/52105933/95031842-07a4ea00-06f3-11eb-9022-db54e399da32.gif)|![矢印（赤）](https://user-images.githubusercontent.com/52105933/110228721-b0779f80-7f46-11eb-8cd9-469501beea50.png)|![Hopper_std_G_202010122350_1000_1579](https://user-images.githubusercontent.com/52105933/95936264-218bae80-0e10-11eb-9804-48b9add13747.gif)|

※HopperPyBulletの画像が白いのはご了承下さい。Google Colaboratory上で3Dのpybullet-gymの動きを色つきでキャプチャーする簡便な方法がありませんでした。<BR>
![hopper_60](https://user-images.githubusercontent.com/52105933/95032126-6159e400-06f4-11eb-81b6-0762ce248401.png) ←本来のHopperPyBulletの外観。脳内イメージ補完してください。<BR><BR>

## 概要
ActorとCriticで中間層を共有する構成と、ActorとCriticを分離する構成の2パターンで構築し、比較しています。<br>

本モデルは、行動が連続値を取る環境を対象としています。<BR>
本稿では、OpenAI GymのPendulum、LunarLanderContinuous、BipedalWalker、pybullet-gymのHopperPyBulletを使用しています。<br>
    
また、フレームワークを使用せず、主にnumpyだけでA2Cを構築しています。<br>

※本稿の「A2C」とは、純粋なAdvantage Actor Criticのことであり、分散処理は含みません。<br>

※理論の説明は基本的にしていません。他のリソースを参考にしてください。<br>
&nbsp;&nbsp;ネットや書籍でなかなか明示されておらず、私自身が実装に際し情報収集や理解に不便を感じたものを中心に記載しています。<br><br>

### ActorとCriticで中間層を共有／分離の2パターンで比較
ActorとCriticで中間層を共有する構成と、ActorとCriticを分離する構成の2パターンで構築し、訓練成果を比較しています。
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

gym.Wrapperで定義され（gym.make("hoge")）、上記を満たす環境なら動作しますが、環境毎の仕様や性質（状態や行動の次元数の大きさ、報酬設計など）により、訓練成果が変動します。<br>

本稿の実験においては、以下の環境を使用しています。<br>

| 環境名 | 外観| 状態の次元 | 行動の次元 |1エピソードでの上限ステップ数|目的|
|      :---:     |      :---:      |     :---:     |     :---:      |     :---:     |     :---:     |
|Pendulum|![pendulum_mini](https://user-images.githubusercontent.com/52105933/95575625-fa2c8e80-0a69-11eb-935c-38da16dc8afa.png)|3|1|上限は無し<br>固定で200ステップ/エピソード|バーの直立|
|LunarLanderContinuous|![LunarLanderContinuous_mini](https://user-images.githubusercontent.com/52105933/95575748-352ec200-0a6a-11eb-8494-997ac236f1ae.png)|8|2|1000|ゴール領域に着陸|
|HopperPyBullet|![hopper_mini](https://user-images.githubusercontent.com/52105933/95576384-50e69800-0a6b-11eb-9a18-711c04d690c9.png)|15|3|1000|遠くまで跳ねて前進|
|BipedalWalker|![BipedalWalker_mini](https://user-images.githubusercontent.com/52105933/95576368-4cba7a80-0a6b-11eb-922e-52c584a8915e.png)|24|4|2000|遠くまで2足歩行して前進|

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
G' =  G / σ　（σは1エピソード内全ステップでの割引報酬和の標準偏差）<BR><BR>
- action（行動）はK次元ベクトルであるが、それら各次元は互いに独立であるとする
<BR><BR>
- 共有型も分離型も、中間層の各レイヤーのノード数は、状態の次元数、行動の次元数に応じて決まる<br>
これら次元数が大きければノードも多い<BR><BR>
- NNのパラメーター更新はエピソード毎　＝　1イテレーション/エピソード<br>
1イテレーションで使用する訓練データは、この1エピソードの試行結果<BR><BR>
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

<b>＜共有型において、Actorの損失をAdvantageを介してCritic側に逆伝播させない理由＞</b><br>
actionの推測は、K次元正規分布N(μ, var)からのサンプリングで行われます。<br>
その際、NNの出力値として使用するのはμとvarだけです。価値関数Vは使用しません。<br>
よって、Actorの損失関数の最小化にVを関与させる＝Critic側のV出力ノードを関与させると、（Vを使用しない）actionの推論時における最適なμとvarにはならないです。<BR>
<br>

#### 割引報酬和の標準化

本稿の全実験において、訓練時にモデル内部で割引報酬和の標準化をしています。<BR>
（訓練関数train()の引数で標準化をするかしないかを指定できます）<br>

割引報酬和は、Advantageを介してCriticのlossになります。<br>
使用した4環境（endulum、LunarLanderContinuous、BipedalWalker、HopperPyBullet）いずれも、割引報酬和を標準化しないと、CriticのlossがActorのlossから大きく乖離してしまいます。<br>
特に1エピソードのステップ数が200ステップ固定のPendulumは、訓練開始時のCriticのlossは10数万にもなります（Actorのlossはマイナス数百程度）。<br>
このような状況にならないように、割引報酬和の標準化をしています。<br>
※試しに割引報酬和の標準化をせずに訓練してみましたが、共有型は全滅し、分離型もあまり機能しませんでした。<br>

ただし、本モデルが内部で行う割引報酬和の標準化では、**平均を0にしません。**<BR>
G' =  G / σ　（σは1エピソード内全ステップでの割引報酬和の標準偏差）<BR>
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

| 環境名 | 外観| 状態の次元 | 行動の次元 |1エピソードでの上限ステップ数|目的|
|      :---:     |      :---:      |     :---:     |     :---:      |     :---:     |     :---:     |
|Pendulum|![pendulum_mini](https://user-images.githubusercontent.com/52105933/95575625-fa2c8e80-0a69-11eb-935c-38da16dc8afa.png)|3|1|上限は無し<br>固定で200ステップ/エピソード|バーの直立|

#### 訓練記録の比較


| 訓練成果（以下の動画は分離型での訓練によるもの）<br>バーは途中から直立を維持 | 
|     :---:      |
|![Pendulum_sep_202010010221](https://user-images.githubusercontent.com/52105933/95010122-d6c8a480-0661-11eb-8703-64dcad0b2d35.gif)|

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

| 環境名 | 外観| 状態の次元 | 行動の次元 |1エピソードでの上限ステップ数|目的|
|      :---:     |      :---:      |     :---:     |     :---:      |     :---:     |     :---:     |
|LunarLanderContinuous|![LunarLanderContinuous_mini](https://user-images.githubusercontent.com/52105933/95575748-352ec200-0a6a-11eb-8494-997ac236f1ae.png)|8|2|1000|ゴール領域に着陸|

#### 訓練記録の比較


| 訓練成果（以下の動画は分離型での訓練によるもの）<br>ゴール内に水平を保って着陸 | 
|     :---:      |
|![LunarLanderc_step_202010022143_2_138s](https://user-images.githubusercontent.com/52105933/95009767-7c2e4900-065f-11eb-8182-a271390bf9df.gif)|

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

共有型・分離型とも、ある程度のscoreに達すると頭打ちになってしまうのは、他の環境にも共通している現象ですので、後でまとめて記述します。

<br>

### HopperPyBullet

| 環境名 | 外観| 状態の次元 | 行動の次元 |1エピソードでの上限ステップ数|目的|
|      :---:     |      :---:      |     :---:     |     :---:      |     :---:     |     :---:     |
|HopperPyBullet|![hopper_mini](https://user-images.githubusercontent.com/52105933/95576384-50e69800-0a6b-11eb-9a18-711c04d690c9.png)|15|3|1000|遠くまで跳ねて前進|

#### 訓練記録の比較


| 訓練成果（以下の動画は共有型での訓練によるもの）<br>完璧な歩様で上限1000ステップまで前進 | 
|     :---:      |
|![Hopper_std_G_202010122350_1000_1579](https://user-images.githubusercontent.com/52105933/95936264-218bae80-0e10-11eb-9804-48b9add13747.gif)|

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
他の環境にも共通している現象ですので、後でまとめて記述します。

<br>

### BipedalWalker

| 環境名 | 外観| 状態の次元 | 行動の次元 |1エピソードでの上限ステップ数|目的|
|      :---:     |      :---:      |     :---:     |     :---:      |     :---:     |     :---:     |
|BipedalWalker|![BipedalWalker_mini](https://user-images.githubusercontent.com/52105933/95576368-4cba7a80-0a6b-11eb-922e-52c584a8915e.png)|24|4|2000|遠くまで2足歩行して前進|

#### 報酬設計の変更　～ gym.Wrapperのサブクラス

BipedalWalkerは、エピソード終端での失敗時（転倒時）、-100というとても大きな絶対値の即時報酬を返してエピソードを終了します。<br>
このように一連の訓練データ中に数値規模が著しく大きい即時報酬がポツンとある場合、それは実質”外れ値”となります。<br>
また、他のエピソード途中のステップの即時報酬値が相対的に小さくなってしまいます。<br>
よって、gym.Wrapperのサブクラスを作成し、以下のような即時報酬を返すようにしました。
|事象|即時報酬|
|      :---:     |      :---:      | 
|エピソード途中のステップ|オリジナルのgym.Wrapperのrewardと同じ|
|エピソード終端　成功時<BR>（ゴールに到達した）|+1|
|エピソード終端　失敗時<BR>（ゴールに到達しなかった）|-1|

#### BipedalWalkerの特性

BipedalWalkerには、（同類の例えばHopperPyBulletと比べて）以下のような特性があります。<br>
- ステップ数が訓練の度合いに比例しない（全く未訓練でいきなり上限2000ステップまで行ったりする）<br>
- ある程度訓練が進むまで、scoreとステップ数はむしろ反比例的（ステップ数が伸びるほどscoreは下がる）<br>
従って、**BipedalWalkerの訓練結果評価指標は、scoreとステップ数の両方**とします。<br>
<b>「scoreとステップ数がともに連動して伸びているか」</b>を見ることにします。

#### 訓練記録の比較


| 訓練成果（以下の動画は共有型での訓練によるもの）<br>少しだけ何とか前進 | 
|     :---:      |
|![BipedalWalker_202103071440_436s_30r](https://user-images.githubusercontent.com/52105933/110274830-20ebf280-8013-11eb-9feb-94b3c9a4dc7a.gif)|

<br>

![共有型と分離型の比較_BipedalWalker_70](https://user-images.githubusercontent.com/52105933/110276900-7fb36b00-8017-11eb-9de1-a5c2fa98bac4.png)

| 共有型/分離型 | 稼得score合計 |稼得score最大値（その時のステップ数）|
|     :---:      |     :---:      |     :---:      |
|共有型|13万|65（684）|
|分離型|-50万|37（380）|

#### まとめ・考察など

**共有型と分離型双方、あまりよく訓練されませんでした。**<br>
共有型は当初からscoreが緩やかに伸びていましたが、途中から停滞してしまいました。<br>
分離型は終盤までscoreは変化せず、終盤から緩やかに伸びました。<br>
「scoreとステップ数がともに連動して伸びているか」については、共有型と分離型双方、若干の改善が見られるものの、**基本的に「ステップ数が大きいほどscoreは小さく、ステップ数が小さいほどscoreは大きい」まま**でした。<br>

訓練未熟の状態では1ステップで得られる即時報酬はマイナスの場合が多く、1ステップの動きを改善する（＝1ステップでの即時報酬を大きなプラスにする）より前に、「傷口が深くなる前に早めにコケてscoreを維持する」ことを学んでしまったのかもしれません。<br>
<br><br>


### 全体のまとめ・考察など

大まかには、以下の現象となりました。<br>
「稼得scoreの絶対量」と「稼得scoreが伸びるスピード」において・・・
- **共有型は概ね分離型より良く機能した。**<br>
- 分離型も悪かったわけではなく、そこそこ機能した。<br>
- 共有型も分離型も、ある程度のところでscoreが伸びなくなった<br>

「共有型も分離型も、ある程度のところでscoreが伸びなくなった」のは、以下のような原因であると考えています。<br>
書いている順番に意味はありません。<br>

①訓練データの使い方が悪い<br>

1イテレーションで使用する訓練データは、イテレーション前に1エピソードのみ試行した結果です。<br>
量の少なさも問題ですが、より問題なのは、互いに相関性の強い時系列データをそのまま訓練に使用している、という点です。<br>
いわゆる「更新の分散」が起こり、訓練が不安定になるはずです。<br>
本稿の実装に着手したのは随分前で、この件に対して、あまり認識はしていませんでした。<br>
もし今実装するとしたら、決してこのようにはせず、エポック毎に2万件くらいのTrajectoryを収集して、各イテレーションではそこからランダム抽出する、という方法を取ると思います。<br>

②高scoreの方策に凝り固まって抜け出せない

ある方策で高scoreを記録するとその方策に凝り固まってしまう、ということが起こっていると考えられます。<br>
これは、「探索（exploration）」をさせればよく、いわゆる方策エントロピー補正項をLossに組み込むことで解決できる可能性があります。<br>

③共有型のCriticのlossの重み係数が不適切<br>

モデル全体のloss = Actorのloss + 重み係数 * Criticのloss <br>
本稿の全実験において、この重み係数をあえて1.0に統一しています。異なる環境間で比較するためです。<br>
が、本来はそれぞれの環境での適切な重み係数というのがあると思います。<br>

<BR><br>

## 実行確認環境と実行の方法

### 実行確認環境

以下の環境での実行を確認しました。<br>

- numpy 1.19.1
- gym 0.17.2
- box2d-py
- pybullet-gym

#### インストール

gymとbox2d-pyのインストール
```
pip install gym
pip install box2d-py
```
pybullet-gymのインストールは、[リポジトリ](https://github.com/benelot/pybullet-gym)のREADME記載の通りに行ってください。<br>
ここにも転記しておきます。
```
pip install pybullet
git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip install -e .
```

### 実行の方法

訓練済モデルの使用、訓練と推論の具体的な方法は、./trained_params/使い方.txtを参照してください。<br>
同じ場所に、Pendulumでの訓練済みパラメーターのpickleファイルがあります。<br>

<br>

## ディレクトリ構成
Planner_share.py<BR>
Planner_separate.py<BR>
common/<br>
&nbsp;└funcs.py<br>
&nbsp;└layers.py<br>
&nbsp;└optimizers.py<br>
&nbsp;└env_wrappers.py<br>
trained_params/<br>
&nbsp;└（訓練済パラメーターのpickleファイル）<br>
-----------------------------------------------------------------------------------------------------<br>
- Planner_share.py：共有型モデル本体。中身はclass Planner_share です。モデルを動かすにはcommonフォルダが必要です。
- Planner_separate.py：分離型モデル本体。中身はclass Planner_separate です。モデルを動かすにはcommonフォルダが必要です。
<br><br>

## モデルの構成
Planner_share.pyのclass Planner_share、Planner_separate.pyのclass Planner_separateが、モデルの実体です。
<br><br>

![モデルの構成_70](https://user-images.githubusercontent.com/52105933/95835589-60225a00-0d79-11eb-8dd9-87f8cd0fe3c1.png)

このclass Planner_xx をアプリケーション内でインスタンス化して、環境の訓練やPlayといったpublicインターフェースを呼び出す、という使い方をします。
```
#モデルのインポート　以下のどちらか 
from Planner_share import * #モデル本体　共有型の場合
from Planner_separate import * #モデル本体　分離型の場合

#Pendulumの環境の活性化　ここではPendulumを使用
env = gym.make("Pendulum-v0")
  
#モデルのインスタンスを生成　以下のどちらか 
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
<br>

### class Planner_share、class Planner_separate　のpublicインターフェース一覧
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

<br>

## その他

本リポジトリは、pybullet-gymのオープンソースのプログラムを利用しており、以下の条件でライセンスされています。<BR>
```

MIT License

Copyright (c) 2018 Benjamin Ellenberger

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

<br><br>


※本リポジトリに公開しているプログラムやデータ、リンク先の情報の利用によって生じたいかなる損害の責任も負いません。これらの利用は、利用者の責任において行ってください。