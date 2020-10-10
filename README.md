# A2C（Advantage Actor Critic）<br>ActorとCriticで中間層を共有／分離の2パターンを構築し比較<br>（フレームワーク不使用）

<br>フレームワークを使用せず、numpyだけでA2Cを構築しました。<br>
ActorとCriticで中間層を共有する構成と、ActorとCriticを分離する構成の2パターンで構築し、比較しています。<br>
環境としては、OpenAI GymのPendulum、LunarLanderContinuous、BipedalWalker、pybullet-gymのHopperPyBulletを使用しています。<br>
<BR>
| LunarLanderContinuous　未訓練モデルでPlay<br>枠外に飛去またはゴール外に落下 | LunarLanderContinuous　訓練済モデルでPlay<br>ゴール内に水平を保って着陸 |
|      :---:       |     :---:      |
|![LunarLanderc_beginner](https://user-images.githubusercontent.com/52105933/95009756-6882e280-065f-11eb-8d15-51e64f56b7b6.gif)|![LunarLanderc_step_202010022143_2_138s](https://user-images.githubusercontent.com/52105933/95009767-7c2e4900-065f-11eb-8182-a271390bf9df.gif)|

| BipedalWalker　未訓練モデルでPlay<br>すぐ転倒し前に進めない | BipedalWalker　訓練済モデルでPlay<br>拙いながら何とか前進 |
|      :---:       |     :---:      |
|![BipedalWalker_beginner_66s](https://user-images.githubusercontent.com/52105933/95009942-9c123c80-0660-11eb-9cb1-b5ee0a2a90f7.gif)|![BipedalWalker_202010011308](https://user-images.githubusercontent.com/52105933/95009953-b1876680-0660-11eb-9865-939f1559bc85.gif)|

| Pendulum　未訓練モデルでPlay<br>バーは一向に立たない | Pendulum　訓練済モデルでPlay<br>バーは途中から直立を維持 |
|      :---:       |     :---:      |
|![Pendulum_sep_biginner](https://user-images.githubusercontent.com/52105933/95010103-b3055e80-0661-11eb-9969-15166ff55da4.gif)|![Pendulum_sep_202010010221](https://user-images.githubusercontent.com/52105933/95010122-d6c8a480-0661-11eb-8703-64dcad0b2d35.gif)|

| HopperPyBullet　未訓練モデルでPlay<br>すぐ転倒し前に進めない | HopperPyBullet　訓練済モデルでPlay<br>上限1000ステップまで転倒せず前進 |
|      :---:       |     :---:      |
|![HopperPyBullet_test](https://user-images.githubusercontent.com/52105933/95031842-07a4ea00-06f3-11eb-9022-db54e399da32.gif)|![Hopper_sep_202010061525_1000_1313](https://user-images.githubusercontent.com/52105933/95550591-1f0e0b00-0a44-11eb-9645-6a2fc253c6f5.gif)|

※HopperPyBulletの画像が白いのはご了承下さい。Google Colaboratory上で3Dのpybullet-gymの動きを色つきでキャプチャーする簡便な方法がありませんでした。<BR>
![hopper_60](https://user-images.githubusercontent.com/52105933/95032126-6159e400-06f4-11eb-81b6-0762ce248401.png) ←本来のHopperPyBulletの外観。脳内イメージ補完してください。<BR><BR>

## 概要
フレームワークを使用せず、numpyだけでA2Cを構築しました。<br>
ActorとCriticで中間層を共有する構成と、ActorとCriticを分離する構成の2パターンで構築し、比較しています。<br>
本モデルは、状態、行動とも連続値を取る環境を対象としています。<BR>
本稿では、OpenAI GymのPendulum、LunarLanderContinuous、BipedalWalker、pybullet-gymのHopperPyBulletを使用しています。<br>
※本稿の「A2C」とは、純粋なAdvantage Actor Criticのことであり、分散処理は含みません。<br>
※理論や実装のヒント、NNの構成については、以下2つの良書を参考にしました。<BR>
　・機械学習スタートアップシリーズ Pythonで学ぶ強化学習 入門から実践まで (KS情報科学専門書) <BR>
　・現場で使える！Python深層強化学習入門 強化学習と深層学習による探索と制御 (AI & TECHNOLOGY) <br><br>

###  フレームワークを使用せずnumpyだけで実装
フレームワークを使用せずにnumpyだけで実装しています。<br>
Actorの損失、誤差逆伝播、その他諸々を自力で0から実装しています。
<br><br>
    
###  ActorとCriticで中間層を共有／分離の2パターンで比較
ActorとCriticで中間層を共有する構成と、ActorとCriticを分離する構成の2パターンで構築し、比較しています。
<br><br>
    
## ActorとCriticで中間層を共有／分離の2パターンの構成

###  両パターンの構成図
ActorとCriticで中間層を共有・・・「共有型」<br>
ActorとCriticを分離・・・「分離型」<br>
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

###  前提

- Actor側のメソッドは方策勾配法（Policy Gradient）<br>
ガウス方策を採用
- Actorの損失関数にAdvantageを使用<br>
Q関数（行動価値関数）を割引報酬和で近似（REINFORCEアルゴリズム）
- Critic側の損失関数は平均2乗和誤差<br>
教師信号は割引報酬和とする
- action（行動）はK次元ベクトルであるが、それら各次元は互いに独立であるとする
- 共有型も分離型も、中間層の各レイヤーのノード数は、状態の次元数、行動の次元数に応じて決まる<br>
これら次元数が大きければノードも多い
- NNのパラメーター更新はエピソード毎　＝　1イテレーション/エピソード
- （共有型のみ）全体の損失 = Actorの損失 + 重み係数 × Criticの損失　とする<br>
本稿の全実験において、この重み係数を1.0に統一する（比較のため）
- 使用する環境は以下の通り

| 環境名 | 外観| 状態の次元 | 行動の次元 |1エピソードでの上限ステップ数|
|      :---:     |      :---:      |     :---:      |     :---:      |     :---:      |
|Pendulum|![pendulum_mini](https://user-images.githubusercontent.com/52105933/95575625-fa2c8e80-0a69-11eb-935c-38da16dc8afa.png)|3|1|上限は無く固定で200ステップ/エピソード|
|LunarLanderContinuous|![LunarLanderContinuous_mini](https://user-images.githubusercontent.com/52105933/95575748-352ec200-0a6a-11eb-8494-997ac236f1ae.png)|8|2|1000|
|HopperPyBullet|![hopper_mini](https://user-images.githubusercontent.com/52105933/95576384-50e69800-0a6b-11eb-9a18-711c04d690c9.png)|15|3|1000|
|BipedalWalker|![BipedalWalker_mini](https://user-images.githubusercontent.com/52105933/95576368-4cba7a80-0a6b-11eb-922e-52c584a8915e.png)|24|4|2000|

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

![loss_actor_計算グラフ2_75](https://user-images.githubusercontent.com/52105933/95654330-565ee380-0b3a-11eb-9de6-5a9e9721a53b.png)

<b>※共有型において、なぜActorの損失をAdvantageを介してCritic側に逆伝播させないのか</b><br>
感覚的な説明しかできませんが・・・<br>
actionの推測は、K次元正規分布N(μ, var)からのサンプリングで行われます。<br>
その際、NNの出力値として使用するのはμとvarだけです。価値関数Vは使用しません。<BR>
よって、Actorの損失関数の最小化にCritic側のV出力ノードを”協力”させるわけにはいかないのです。<br>
・・・と思います。

<br><br>

## ActorとCriticで中間層を共有／分離の2パターンの比較

Pendulum、LunarLanderContinuous、BipedalWalker、HopperPyBullet　の4環境を、<br>
共有型、分離型それぞれで訓練しました。<br>
「score（スコア）」とは、1エピソードで得られた報酬の単純合計のことです。<br>
NNのパラメーター更新はエピソード毎　＝　1イテレーション/エピソード　です。<BR>
下記グラフの横軸は全てエピソードです。
<br><br>

### Pendulum

| 状態の次元 | 行動の次元 |1エピソードでの上限ステップ数|
|     :---:      |     :---:      |     :---:      |
|3|1|上限は無く固定で200ステップ/エピソード|

#### 訓練記録の比較
<br>

![共有型と分離型の比較_Pendulum_70](https://user-images.githubusercontent.com/52105933/95016928-37230a80-0691-11eb-846f-a97f875ce120.png)

| 共有型/分離型 | 稼得score合計 |稼得score最大値|
|     :---:      |     :---:      |     :---:      |
|共有型|-5006万|-595|
|分離型|-3357万|-0.33|

#### まとめ・考察など

共有型はほぼ機能しませんでした。<br>
一方で、分離型ではバーの直立を維持するのに十分なscoreを獲得するまで訓練することができました。

**共有型が崩壊したのは、「Pendulumの仕様により莫大なCriticのlossが半永続的に発生し、マルチタスク学習にならなかった」から**だと思います。<BR>

- Pendulumの仕様により莫大なCriticのlossが半永続的に発生<br>
&nbsp;<br>
Pendulumの仕様の特徴は、「1エピソードのステップ数は常に固定で200」です。これにより、特に訓練初期は、割引報酬和は大きなマイナス数値になります。<br>
そして、Criticの損失関数は平均”2乗”誤差であり、教師信号は割引報酬和です。<br>
従って、Criticのlossは10数万という莫大な数値になります。<br>
&nbsp;<br>
さらに、Pendulumは1エピソードのステップ数が固定で200なので、Agentの行動によってステップ数が減ってマイナスに大きい割引報酬和が緩和される、ということもありません。<br>
つまり、「莫大なCriticのloss」は訓練初期から半永続的なものになります。<br>

- マルチタスク学習にならなかった<br>
&nbsp;<br>
分離型の場合は「莫大なCriticのloss」はCritic側にとどまる話なので、上記グラフのように、それでも地道に訓練は進み、scoreを伸ばせます。<br>
（マイナスに大きい割引報酬和はadvantageを介してActorのlossにもなるが、この場合advantageはマイナスになり数値としてはマイナス数百程度）<br>
&nbsp;<br>
しかし**共有型の場合、この莫大なCriticのlossはモデル全体のlossとなり（Actorのlossと合算）、NNの共有部分はこの莫大なCriticのlossを丸かぶり**せねばなりません。<br>
共有型はいわゆる「マルチタスク学習」です。「方策の最適化（Actor側）」と「価値関数の最適化（Critic側）」の2つのタスクを同時にこなします。<br>
が、Criticのloss（10数万）がActorのloss（マイナス数百程度）に比してケタ違いに大きいため、共有部分では「方策の最適化（Actor側）」の方は為されなかったのだと思います。<br>
&nbsp;<br>
ちなみに、共有型のモデル全体のlossの計算式<br>
全体の損失 = Actorの損失 + 重み係数 * Criticの損失<br>
の重み係数を0.1などにしてみましたが、効果ありませんでした。文字通り「ケタが違う」のだと思います。<br>
これ以上重み係数を小さくすると、価値関数の最適化、というタスクの方が果たせなくなると思ったので、これ以上小さくしませんでした。<br>

結局、少なくとも以下のようなことが言えると思います。<br>
・CriticのlossがActorのlossより遙かに大きい場合、共有型を使用するべきではない<br>
　（Pendulumのように）上記の大きな乖離が解消されにくい場合はなおさら

<br>

### LunarLanderContinuous

| 状態の次元 | 行動の次元 |1エピソードでの上限ステップ数|
|     :---:      |     :---:      |     :---:      |
|8|2|1000|

#### 訓練記録の比較
<br>

![共有型と分離型の比較_LunarLanderConti_70](https://user-images.githubusercontent.com/52105933/95654889-17cb2800-0b3e-11eb-930f-932c8a17e664.png)

| 共有型/分離型 | 稼得score合計 |稼得score最大値|
|     :---:      |     :---:      |     :---:      |
|共有型|-312万|76|
|分離型|-93万|244|

#### まとめ・考察など

共有型ははじめ3分の1ほどは稼得scoreに成長が見られたものの0付近で停滞し、そのまま終わりました。<br>
分離型では共有型よりかなり高いscoreを獲得するまで訓練することができました。<br>
分離型の方が相対的に高いパフォーマンスを発揮しました。<br>

Pendulumと同様、訓練初期は割引報酬和Gの数値が高く、Criticのlossは10数万という莫大な数値になります。<br>
それでも**Pendulumの時のように共有型が崩壊しなかったのは、1エピソードのステップ数が（Pendulumと異なり）固定ではないから**、と思います。<br>

Agentの行動次第でステップ数に変化がつくので、訓練初期は毎エピソード「必ず」累積報酬和が大きなマイナスの数値になるとは限らず、累積報酬和が大きなマイナスの数値になる<b>「傾向」</b>がある、としかならないです。<br>
よって、訓練初期であっても共有型においてNN共有部分が丸かぶりするCriticのlossは「必ず」莫大な数値になるわけではなく、そうなる<b>「傾向」</b>があるだけ、となります。<br>
共有型のlossのグラフはまさにそうなっています。<br>
ここがPendulumとの違いとなります。<br>

ただこの<b>「NN共有部分が丸かぶりするCriticのlossは莫大な数値になる傾向がある」は分離型に対してハンデ</b>であることは明らかで、共有型は分離型より劣ってしまうのだと思います。<br>

その他の原因（NN表現力不足、Criticのlossの重み係数が不適切、など）については、他の環境と共通しているので、後でまとめて記述します。

<br>

### HopperPyBullet

| 状態の次元 | 行動の次元 |1エピソードでの上限ステップ数|
|     :---:      |     :---:      |     :---:      |
|15|3|1000|

#### 訓練記録の比較
<br>

![共有型と分離型の比較_HopperPyBullet_70](https://user-images.githubusercontent.com/52105933/95289327-6f9b3200-08a5-11eb-98cd-2e449ebab982.png)

| 共有型/分離型 | 稼得score合計 |稼得score最大値|
|     :---:      |     :---:      |     :---:      |
|共有型|862万|901|
|分離型|2197万|1340|

#### まとめ・考察など

共有型は、ほんの初期だけ訓練が進み、以降は訓練が停滞しているように見えます。<br>
一方で、分離型は、2度スランプがあるものの、scoreは時間の経過とともに伸びてます（ただしペースは緩やか）。<br>
稼得scoreの大小比較でも、分離型が勝っています（合計では、共有型の862万に対して分離型は2197万）。<br>

共有型でほんの初期だけ訓練が為されるが以降は止まってしまう原因として、NN共有部分の表現力不足の可能性を考えています。<br>
共有型の共有部分（中間層全体）と、分離型のActorのNNとCriticのNNそれぞれの中間層のノード数やレイヤー数は、同じにしてあります（比較のため）。<BR>
**分離型では十分な構成でも、共有型では不十分、ということが起こってる**のでは、と考えています。<BR>

この表現力不足という原因や、その他の考えられる原因については、他の環境と共通しているので、後でまとめて記述します。

<br>

### BipedalWalker

| 状態の次元 | 行動の次元 |1エピソードでの上限ステップ数|
|     :---:      |     :---:      |     :---:      |
|24|4|2000|

#### 訓練記録の比較
<br>

![共有型と分離型の比較_BipedalWalker_70](https://user-images.githubusercontent.com/52105933/95461853-ac544f80-09b1-11eb-8fa8-0bd155112f01.png)

| 共有型/分離型 | 稼得score合計 |稼得score最大値|
|     :---:      |     :---:      |     :---:      |
|共有型|-207万|-95|
|分離型|-252万|-96|

#### まとめ・考察など

共有型と分離型の違いよりも、そもそも訓練されたのかどうかがよく分からない結果でした。<br>
イテレーションを1.5万回繰り返しても、稼得scoreの最高値にほぼ変化は無かったです。<br>
他の環境と比較するとこのBipedalWalkerが最も状態の次元数（24）と行動の次元数（4）が高く、**共有型・分離型双方、NN中間層の表現力が不足している可能性**があります。<BR>
状態の次元数と行動の次元数が増えるとNN中間層のノードも増えるような作りにしていますが、追いついていないのかもしれません。

<br>

### 全体のまとめ・考察など

大まかには、以下の現象となりました。<br>
「稼得scoreの絶対量」と「稼得scoreが伸びるスピード」において・・・
- **共有型はほとんど機能しなかった。**
- 分離型もそれほど良かったわけではない。<br>
スピードが緩やか、ある時点で訓練の進行が停滞、訓練そのものが為されない、など。

以下のような原因であると考えています。<br>
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

②確率的サンプリング～割引報酬和が大きく変動<br>

方策勾配法・ガウス方策を採用しており、Actor側から出力した平均μと分散varの正規分布から、ランダムにactionをサンプリングします。<br>

仮に、2つのAgentが全く同一の状況（同一state、同一時刻t、同一のNNのパラメーター）にあったとしても、actionが確率に基づくサンプリングなので、エピソード終端までの割引報酬和は互いに異なるはずです。<br>
増してや、state以外が同一でなければ、割引報酬和はもっと異なるはずです。<br>
つまり、同一の入力stateに対してであっても、Critic側の損失関数の教師信号（＝割引報酬和）が異なり、共有型の場合は、Actorの損失関数に与えるadvantage（＝割引報酬和－価値関数）も異なる、ということになります。<br>

このように、Critic側の損失関数の教師信号、共有型の場合はActorの損失関数に与えるadvantageがブレるので、訓練が不安定になり進まなくなる、ということになると思います。<br>

③共有型・分離型ともに表現力不足<br>

共有型・分離型ともに、状態の次元数と行動の次元数が増えるとNN中間層のノードも増えるような作りにしています。<br>
が、状態と行動の次元数がともに最高のBipedalWalkerでは共有型・分離型ともに訓練がほとんどされず、次点のHopperPyBulletでは分離型ですら訓練スピードがとても緩やかであるところを見ると、
ノードの増やし方が不十分なのかもしれません。<br>

④共有型の表現力不足<br>

- 共有部分の表現力不足<br>
共有型はいわゆる「マルチタスク学習」です。「方策の最適化（Actor側）」と「価値関数の最適化（Critic側）」の2つのタスクを同時にこなします。<br>
そして、共有型の中間層である共有部分は、「方策（正確にはμとlog(var)）」と「価値関数」を出力するための主要部分です。<br>
つまり、**共有部分は、「方策」と「価値関数」双方に通用する”汎用的な”関数である必要**があります。<br>
また、この部分の構成は、比較のため、分離型のActorとCriticのNNの中間層と同じ構成にしています。<BR>
一方で、分離型のActorの中間層は「方策」だけを、Criticの中間層は「価値関数」だけを出力するための主要部分となります。<br>
つまり、**共有型の中間層（共有部分）は、分離型の各々の中間層よりも重い役目を背負わされているにもかかわらず、それら分離型の各々の中間層と同じレイヤー数・ノード数しか与えられていない**、ということになります。<br>

- 出力部分の表現力不足<br>
”汎用的”である共有部分の出力を、Actorの出力層が受け取って、方策（正確にはμとlog(var)）を出力します。つまり”単一の目的に特化”しています。<br>
同様に、Criticの出力層は、価値関数の出力という”単一の目的に特化”しています。<br>
<b>”汎用的である物”を”単一の目的に特化した物”に変換するのがたったの1レイヤー（出力層）だけ、というのは、やはり足りない</b>のでは、と思います。<br>

<<新しい共有型（案）>><br>
<!--![NN概念図_新案_70](https://user-images.githubusercontent.com/52105933/95455477-e79e5080-09a8-11eb-9b0b-99f9e0f267cd.png)-->
![NN概念図_新案_60](https://user-images.githubusercontent.com/52105933/95455942-8b87fc00-09a9-11eb-9ddb-5c8107d129b8.png)

⑤共有型のCriticのlossの重み係数が不適切<br>

モデル全体のloss = Actorのloss + 重み係数 * Criticのloss <br>
本稿の全実験において、この重み係数をあえて1.0に統一しています。異なる環境間で比較するためです。<br>
しかし、Criticのlossが各環境間で大きく異なることから、本来はそれぞれ適切な重み係数があります。<br>

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

![モデルの構成_80](https://user-images.githubusercontent.com/52105933/95302253-a761a400-08bc-11eb-89dc-c8ad9fcac956.png)

このclass Planner_xx をアプリケーション内でインスタンス化して、環境の訓練やPlayといったpublicインターフェースを呼び出す、という使い方をします。
```
#モデルのインポート 
from Planner_share import * #モデル本体　共有型の場合
from Planner_separate import * #モデル本体　分離型の場合

#Pendulumの環境の活性化
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

<br>

※本リポジトリに公開しているプログラムやデータ、リンク先の情報の利用によって生じたいかなる損害の責任も負いません。これらの利用は、利用者の責任において行ってください。