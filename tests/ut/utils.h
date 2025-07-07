// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <set>
#include <vector>

#include "catch2/generators/catch_generators.hpp"
#include "knowhere/binaryset.h"
#include "knowhere/dataset.h"
#include "knowhere/object.h"
#include "knowhere/range_util.h"
#include "knowhere/version.h"

constexpr int64_t kSeed = 42;
using IdDisPair = std::pair<int64_t, float>;
struct DisPairLess {
    bool
    operator()(const IdDisPair& p1, const IdDisPair& p2) {
        return p1.second < p2.second;
    }
};

inline knowhere::DataSetPtr
GenDataSet(int rows, int dim, int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<> distrib(0.0, 100.0);
    float* ts = new float[rows * dim];
    for (int i = 0; i < rows * dim; ++i) ts[i] = distrib(rng);
    auto ds = knowhere::GenDataSet(rows, dim, ts);
    ds->SetIsOwner(true);
    return ds;
}

inline knowhere::DataSetPtr
GentmpQueryDataSet() {
    int rows = 1;
    int dim = 768;
    float* ts = new float[rows * dim]{
        0.03061383031308651,     0.0071998098865151405,  -0.013125288300216198,   -0.0006471606902778149,
        0.05688357725739479,     0.02387615665793419,    0.021125124767422676,    0.06313449889421463,
        -0.027054298669099808,   -0.019589824602007866,  -0.031362030655145645,   -0.03259752318263054,
        -0.0268077589571476,     0.015231696888804436,   -0.00905217882245779,    0.03489718586206436,
        0.07895240187644958,     0.01249347161501646,    -0.02310025691986084,    0.0404418520629406,
        -0.016765842214226723,   0.008327941410243511,   0.026853011921048164,    6.355466666718712e-06,
        0.03813110664486885,     -0.0014819614589214325, -0.020841557532548904,   0.08983741700649261,
        -0.04415253922343254,    0.03590049222111702,    0.047885045409202576,    -0.02047399803996086,
        0.041442856192588806,    -0.002278294414281845,  0.020530689507722855,    0.023424562066793442,
        0.010633689351379871,    0.02000165730714798,    0.006265364121645689,    -0.029999203979969025,
        -0.021607130765914917,   0.012852450832724571,   -0.012021203525364399,   -0.012455925345420837,
        0.0015188270481303334,   0.04036907106637955,    0.008334082551300526,    -0.042568907141685486,
        -0.024544065818190575,   0.003340494353324175,   -0.050516653805971146,   0.004009816329926252,
        -0.045146431773900986,   0.03405379131436348,    -0.03790578991174698,    0.0383414588868618,
        -0.010940775275230408,   -0.013698880560696125,  -0.03465269133448601,    -0.02454962395131588,
        0.02840673178434372,     0.020787985995411873,   0.028429977595806122,    0.01981818862259388,
        0.017528878524899483,    0.005236390046775341,   -0.012072255834937096,   -0.005825310945510864,
        -0.021052280440926552,   0.03162899985909462,    0.020182399079203606,    0.0006348890019580722,
        0.04760643467307091,     0.004241678863763809,   0.0241082850843668,      -0.042061712592840195,
        -0.09056565165519714,    0.006893676705658436,   0.014782018028199673,    0.06001875549554825,
        0.04033829644322395,     0.013768398202955723,   0.010260706767439842,    -0.006639128550887108,
        -0.016835160553455353,   0.042963746935129166,   -0.04955698922276497,    -0.04711820185184479,
        -0.019551977515220642,   0.03852896764874458,    -0.027243031188845634,   0.0018310287268832326,
        -0.00019829820666927844, 0.03929769992828369,    -0.028552085161209106,   -0.06422336399555206,
        0.06610264629125595,     -0.0008962817373685539, 0.0377340205013752,      -0.010352588258683681,
        0.005908533465117216,    0.006733352318406105,   0.01495711226016283,     0.04381163790822029,
        -0.027234282344579697,   -0.014051408506929874,  -0.021617937833070755,   -0.004390393383800983,
        -0.01563849113881588,    0.011666124686598778,   -0.017715875059366226,   -0.00632396899163723,
        -0.016907328739762306,   -0.03542983904480934,   -0.02853851020336151,    0.07359800487756729,
        0.050509728491306305,    -0.05325537174940109,   0.043070290237665176,    -0.015222898684442043,
        0.04737509414553642,     0.020655907690525055,   -0.002531748265028,      0.030838126316666603,
        -0.016124827787280083,   0.0034414527472108603,  -0.04334395378828049,    0.05215878784656525,
        -0.03939354792237282,    -0.07271326333284378,   -0.002411498688161373,   0.02136547490954399,
        0.03876471519470215,     -0.03721318021416664,   0.030979692935943604,    0.008069554343819618,
        0.00715482234954834,     0.0006815759115852416,  -0.009731719270348549,   -0.043943341821432114,
        -0.02172740176320076,    -0.03246520459651947,   -0.022819461300969124,   -0.09423496574163437,
        0.032157059758901596,    -0.064773328602314,     0.0005611651577055454,   -0.04180465638637543,
        -0.019406575709581375,   0.01693081669509411,    0.04114426299929619,     0.033227816224098206,
        0.034403376281261444,    -0.018116550520062447,  -0.023506812751293182,   -0.034303728491067886,
        0.02442287467420101,     -0.007848135195672512,  0.010553318075835705,    -0.02751600742340088,
        0.008790438063442707,    -0.011287887580692768,  -0.020312273874878883,   -0.008337661623954773,
        -0.0021520089358091354,  0.011251917108893394,   0.05870800092816353,     0.01180744357407093,
        -0.07549912482500076,    0.021665489301085472,   -0.034775227308273315,   -0.06233000382781029,
        0.02809298038482666,     -0.031937289983034134,  0.02690490148961544,     -0.004704828839749098,
        0.05665898695588112,     0.014217773452401161,   -0.019908946007490158,   -0.02831621654331684,
        -0.04874382168054581,    0.0555153489112854,     -0.009260745719075203,   0.022086190059781075,
        0.036764342337846756,    -0.031473349779844284,  0.044692881405353546,    -0.05317959189414978,
        0.031616684049367905,    -0.023651830852031708,  -0.015856198966503143,   -0.022057635709643364,
        0.05647236853837967,     -0.012171745300292969,  0.02536351978778839,     0.02269573137164116,
        -0.0034804462920874357,  -0.002765367738902569,  -0.003817884251475334,   0.049190420657396317,
        0.0014670431846752763,   0.019139185547828674,   -0.014175024814903736,   -0.09015745669603348,
        -0.023404309526085854,   0.020937355235219002,   0.031023982912302017,    0.04402746632695198,
        -0.02598625048995018,    -0.017985275015234947,  -0.04813817888498306,    -0.04455871880054474,
        0.022396858781576157,    -0.009071428328752518,  0.06665191799402237,     -0.04728259518742561,
        -0.017554722726345062,   -0.0605478473007679,    0.06498125195503235,     -0.0003388347977306694,
        0.08058512210845947,     0.019007015973329544,   -0.025889411568641663,   0.008368597365915775,
        -0.04003836587071419,    0.04419384151697159,    0.055642787367105484,    -0.04341700300574303,
        -0.00711837038397789,    0.09560731053352356,    0.02008478157222271,     0.00656443415209651,
        0.07352816313505173,     -0.07799935340881348,   0.04297764599323273,     0.01322876662015915,
        -0.0334172397851944,     -0.04100707918405533,   0.06560944765806198,     -0.03823572024703026,
        -0.02309287153184414,    0.05682414770126343,    0.02436288446187973,     0.013383624143898487,
        0.04945993795990944,     0.023578355088829994,   -0.018384207040071487,   0.027364827692508698,
        -0.03403926268219948,    -0.01214973907917738,   -0.05207224562764168,    0.018106017261743546,
        0.01580376923084259,     -0.03665202483534813,   -0.01739654131233692,    -0.04706992954015732,
        -0.01521807536482811,    0.015296874567866325,   -0.048524096608161926,   0.035645317286252975,
        -0.0005349521525204182,  0.03054364211857319,    -0.013822844251990318,   0.014644992537796497,
        -0.03350089490413666,    -0.02787739969789982,   -0.03519178926944733,    -0.031528111547231674,
        -0.02912112884223461,    0.015571937896311283,   0.017400631681084633,    0.03391614183783531,
        -0.013270942494273186,   0.007399422582238913,   0.05339079350233078,     0.04247869551181793,
        -0.004826883785426617,   -0.002634621923789382,  -0.0195460207760334,     -0.006688355468213558,
        -0.021702496334910393,   0.004918245133012533,   -0.018395639955997467,   -0.06913638859987259,
        -0.005047168582677841,   -0.03153833746910095,   -0.03616216406226158,    -0.013692413456737995,
        -0.025721624493598938,   -0.0002812913153320551, -0.031260229647159576,   -0.06440715491771698,
        0.002156339818611741,    0.012880698777735233,   -0.012504247017204762,   0.012848056852817535,
        0.008069748058915138,    0.06458399444818497,    0.023752646520733833,    -0.021352889016270638,
        -0.014441496692597866,   0.042312007397413254,   -0.0030541049782186747,  -0.019212348386645317,
        -0.02195800095796585,    -0.03784157708287239,   0.015897609293460846,    -0.06006265804171562,
        -0.2309809923171997,     6.626957474509254e-05,  -0.02911427989602089,    -0.035554077476263046,
        0.09582556784152985,     -0.032509155571460724,  -0.024942731484770775,   -0.0022145656403154135,
        -0.04523100331425667,    0.021939553320407867,   0.07204990833997726,     -0.030004482716321945,
        0.0665188580751419,      -0.04323861747980118,   -0.020025139674544334,   0.024580225348472595,
        0.00336257042363286,     -0.07453141361474991,   0.025501523166894913,    -0.014160878956317902,
        -0.012673509307205677,   -0.040933601558208466,  -0.03697960078716278,    0.02755768969655037,
        0.030674751847982407,    -0.00521111860871315,   -0.0701758936047554,     0.016387395560741425,
        -0.05683765187859535,    -0.015127280727028847,  -0.029341718181967735,   0.014272734522819519,
        -0.07529908418655396,    0.05243905633687973,    0.002534562489017844,    0.01497560739517212,
        -0.036881424486637115,   0.008056391961872578,   0.003887156955897808,    -0.00949411652982235,
        -0.028379634022712708,   0.005737796425819397,   -0.03794113174080849,    -0.0290435329079628,
        0.011161615140736103,    0.019747991114854813,   -0.008341486565768719,   -0.06613869220018387,
        0.02310190536081791,     0.047220032662153244,   -0.02781679667532444,    0.017067760229110718,
        -0.026712490245699883,   0.015405683778226376,   -0.0006692555616609752,  0.015928300097584724,
        -0.024827865883708,      0.07550542056560516,    -0.00839878711849451,    0.07579946517944336,
        0.0025273591745644808,   -0.03761471062898636,   -0.057133350521326065,   -0.07473059743642807,
        -0.04624944552779198,    -0.035637591034173965,  -0.03375203534960747,    -0.01457227673381567,
        0.032778311520814896,    -0.022276226431131363,  0.029039010405540466,    0.014100109227001667,
        -0.0318116769194603,     -0.10807974636554718,   0.02066701464354992,     -0.03352028504014015,
        0.027805019170045853,    -0.0695992112159729,    0.042016275227069855,    -0.0014813659945502877,
        0.04636652022600174,     -0.03692528232932091,   0.02553834766149521,     -0.0043641347438097,
        0.022304555401206017,    -0.002061487641185522,  -0.013797029852867126,   0.042958587408065796,
        -0.020431291311979294,   -0.04366222023963928,   0.04923547804355621,     -0.00864187441766262,
        0.01476705726236105,     0.006707340013235807,   -0.005890741944313049,   -0.0037285457365214825,
        0.020691361278295517,    -0.01006381493061781,   -0.019283220171928406,   -0.006072893273085356,
        -0.016222765669226646,   -0.0527765154838562,    -0.07122711092233658,    -0.04734661430120468,
        0.03555653244256973,     -0.09199251979589462,   -0.02302423305809498,    0.027464697137475014,
        0.03694174438714981,     -0.008462139405310154,  -0.02978607639670372,    -0.01083279773592949,
        0.011714974418282509,    -0.05166798084974289,   -0.003316495567560196,   -0.010591259226202965,
        -0.02232508175075054,    0.031110692769289017,   0.01424469891935587,     -0.001062773517332971,
        0.03767486661672592,     0.03826718404889107,    0.020350513979792595,    0.018258506432175636,
        -0.08087877929210663,    0.026078103110194206,   0.0053899940103292465,   0.020365208387374878,
        0.042748283594846725,    -0.013655485585331917,  -0.01760459691286087,    0.025639791041612625,
        0.02462455816566944,     0.030230235308408737,   -0.05705251172184944,    0.00515558198094368,
        -0.00402907095849514,    -0.08731050789356232,   0.016858059912919998,    0.0005042969714850187,
        -0.016204414889216423,   0.026762232184410095,   0.05146149545907974,     0.0005690242978744209,
        0.09249647706747055,     0.060968078672885895,   0.003348062513396144,    -0.020937230437994003,
        0.015918593853712082,    0.048332538455724716,   -0.026901328936219215,   -0.10011421889066696,
        0.03270561993122101,     -0.0202577356249094,    -0.007666769903153181,   -0.03339104354381561,
        0.01088275108486414,     0.00427209259942174,    0.018482031300663948,    -0.04000948369503021,
        0.03002459928393364,     -0.021217819303274155,  -0.023478031158447266,   -0.03703862428665161,
        -0.01743834652006626,    0.05495067313313484,    0.006332543678581715,    0.02700437791645527,
        -0.026343725621700287,   0.05489353463053703,    0.03800613433122635,     -0.01454458199441433,
        -0.04173988476395607,    -0.016677703708410263,  -0.01229583378881216,    0.019972991198301315,
        -0.04039603844285011,    0.02963612601161003,    -0.011291922070086002,   0.011996360495686531,
        0.003242619801312685,    -0.00357280601747334,   0.0044265203177928925,   -0.016383713111281395,
        0.029628071933984756,    -0.009148851037025452,  0.02744463086128235,     -0.06046565994620323,
        -0.0020055228378623724,  -0.01906595565378666,   0.017627310007810593,    -0.024349628016352654,
        -0.051737330853939056,   0.05032177269458771,    -0.0633440613746643,     -0.09675191342830658,
        0.03578495979309082,     0.040094245225191116,   0.04902747645974159,     -0.00546069024130702,
        -0.0036683466751128435,  -0.0057629551738500595, -0.0370730496942997,     0.05117722228169441,
        0.04926210641860962,     -0.02264299988746643,   0.04455113783478737,     0.0069865882396698,
        0.027151884511113167,    -0.02437872439622879,   0.014271868392825127,    -0.037554193288087845,
        -0.06228421628475189,    -0.018250558525323868,  -0.020040836185216904,   -0.09014498442411423,
        0.018913589417934418,    -0.06051245331764221,   0.0008626424823887646,   -0.024558136239647865,
        -0.014885297045111656,   -0.004473309498280287,  0.021180691197514534,    -0.010536218993365765,
        -0.013916797004640102,   0.0009100205497816205,  -0.0005990044446662068,  -0.019013680517673492,
        0.07405122369527817,     -0.03457435220479965,   0.009058552794158459,    -0.00408786628395319,
        -0.0034615134354680777,  -0.004250270314514637,  0.006414138246327639,    0.02818070538341999,
        -0.022083787247538567,   -0.029294930398464203,  -0.06207054853439331,    0.0013836618745699525,
        -0.009792032651603222,   0.038308028131723404,   0.04249422997236252,     -0.06527580320835114,
        0.005118188913911581,    -0.035164348781108856,  0.004080888349562883,    0.012424403801560402,
        0.03505804017186165,     0.036097899079322815,   0.02991628833115101,     0.0010580377420410514,
        0.014124108478426933,    0.039843298494815826,   0.014320049434900284,    -0.0585525780916214,
        0.01685156300663948,     0.028501756489276886,   0.0017988821491599083,   0.01260091457515955,
        0.02201934903860092,     -0.00281665101647377,   -0.032840583473443985,   -0.004197453148663044,
        0.015222705900669098,    0.07050442695617676,    -0.005869908723980188,   -0.013203196227550507,
        0.027042554691433907,    -0.12219750136137009,   -0.03863295912742615,    -0.03470386564731598,
        0.03947051241993904,     0.016696535050868988,   0.02435268461704254,     -0.04940285161137581,
        0.02286597713828087,     0.005619196221232414,   0.034830935299396515,    -0.02758587710559368,
        -0.07054564356803894,    0.02219892106950283,    0.003845999715849757,    0.00920871738344431,
        0.01989295892417431,     0.01719638705253601,    0.018457558006048203,    0.02450539916753769,
        -0.07545213401317596,    -0.001806019339710474,  0.028967618942260742,    0.016957538202404976,
        0.01606903038918972,     0.03193283453583717,    -0.06376537680625916,    -0.015559721738100052,
        0.038615092635154724,    0.0546046644449234,     -0.006574216764420271,   0.03939694166183472,
        -0.07205633819103241,    0.08957364410161972,    -0.009708227589726448,   0.02116353064775467,
        -0.02245514839887619,    0.016851535066962242,   -0.009149275720119476,   -0.001855314476415515,
        -0.009093091823160648,   -0.0018295086920261383, -0.00977898295968771,    -0.03996653109788895,
        0.01565362885594368,     0.012615596875548363,   -0.05960537865757942,    -0.04213897883892059,
        0.03738287463784218,     -0.021984513849020004,  -0.020752986893057823,   -0.016198720782995224,
        0.01441865786910057,     -0.024687955155968666,  0.049459394067525864,    0.03943266347050667,
        -0.0017640339210629463,  0.06514922529459,       0.0019230047473683953,   0.05162918195128441,
        0.03566286340355873,     0.0715370774269104,     0.06387816369533539,     -0.018882496282458305,
        -0.03186655044555664,    0.03642997145652771,    -0.012260755524039268,   -0.021455248817801476,
        0.028088128194212914,    0.03036772832274437,    -0.027815349400043488,   0.04476955160498619,
        0.017352428287267685,    0.035486966371536255,   0.008238441310822964,    0.05192552134394646,
        0.025288306176662445,    -0.0031526510138064623, 0.02283416874706745,     0.012698297388851643,
        -0.005142164416611195,   0.009911996312439442,   -0.0703047588467598,     0.034205541014671326,
        0.036159541457891464,    0.05873197317123413,    -0.007952040061354637,   -0.03600520268082619,
        -0.04463227093219757,    -0.01004759967327118,   -0.06662268191576004,    0.0011964809382334352,
        0.01979464292526245,     -0.032764893025159836,  0.047638438642024994,    -0.017935043200850487,
        -0.005137131083756685,   -0.028848234564065933,  0.008900688961148262,    -0.05920328199863434,
        0.036295074969530106,    0.023443685844540596,   -0.010137336328625679,   0.01015902403742075,
        -0.04273980110883713,    -0.019705496728420258,  0.008499396964907646,    0.0497329942882061,
        0.01638432964682579,     0.02084723487496376,    -0.0033891890197992325,  0.007837122306227684,
        0.013777545653283596,    0.03562392294406891,    -0.012968582101166248,   -0.050731997936964035,
        0.002963655861094594,    -0.04773230105638504,   0.023422369733452797,    0.025467416271567345,
        -0.019111160188913345,   -0.036732714623212814,  -0.03231688588857651,    0.012981674633920193,
        -0.05283690243959427,    0.0019081929931417108,  -0.0487956628203392,     -0.048402685672044754,
        0.04855383560061455,     -0.007924912497401237,  -0.009476667270064354,   0.044452641159296036,
        0.03423532471060753,     -0.016736481338739395,  -0.02013636752963066,    -0.026567941531538963,
        0.013909239321947098,    0.007282797712832689,   -0.00031507943640463054, 0.005588621832430363,
        -0.05656900256872177,    0.04616842791438103,    -0.05558621510863304,    0.04518164321780205,
        -0.0416179895401001,     0.012149999849498272,   -0.012342175468802452,   -0.02597513608634472,
        0.01113603264093399,     0.019442152231931686,   0.026134846732020378,    0.011747360229492188,
        -0.006643462460488081,   0.0029134107753634453,  0.03773132339119911,     0.06419923901557922,
        -0.022199610248208046,   0.05757443606853485,    0.0478934720158577,      -0.008629538118839264,
        0.03992598131299019,     0.037258800119161606,   0.014484317041933537,    -0.029019223526120186,
        -0.0521014966070652,     -0.04059254005551338,   -0.05377747491002083,    0.010425902903079987,
        0.004721113480627537,    0.041265930980443954,   0.00965752825140953,     0.0037332738284021616,
        0.07620251923799515,     -0.0011072144843637943, 0.02594464085996151,     0.022406745702028275,
        0.03137728571891785,     -0.0017859393265098333, -0.004553596954792738,   -0.018533499911427498,
        0.00445273332297802,     0.030588652938604355,   0.020313140004873276,    -0.05472355708479881,
        -0.04321495443582535,    -0.03394661843776703,   0.07675854861736298,     -0.032031115144491196,
        -0.010518872179090977,   0.017957331612706184,   0.045729879289865494,    0.04238613322377205};
    auto ds = knowhere::GenDataSet(rows, dim, ts);
    ds->SetIsOwner(true);
    return ds;
}

inline knowhere::DataSetPtr
CopyDataSet(knowhere::DataSetPtr dataset, const int64_t copy_rows) {
    REQUIRE(!dataset->GetIsSparse());
    auto rows = copy_rows;
    auto dim = dataset->GetDim();
    auto data = dataset->GetTensor();
    float* ts = new float[rows * dim];
    memcpy(ts, data, rows * dim * sizeof(float));
    auto ds = knowhere::GenDataSet(rows, dim, ts);
    ds->SetIsOwner(true);
    return ds;
}

inline knowhere::DataSetPtr
GenBinDataSet(int rows, int dim, int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<> distrib(0.0, 100.0);
    int uint8_num = dim / 8;
    uint8_t* ts = new uint8_t[rows * uint8_num];
    for (int i = 0; i < rows * uint8_num; ++i) ts[i] = (uint8_t)distrib(rng);
    auto ds = knowhere::GenDataSet(rows, dim, ts);
    ds->SetIsOwner(true);
    return ds;
}

inline knowhere::DataSetPtr
CopyBinDataSet(knowhere::DataSetPtr dataset, const int64_t copy_rows) {
    REQUIRE(!dataset->GetIsSparse());
    auto rows = copy_rows;
    auto dim = dataset->GetDim();
    auto data = dataset->GetTensor();
    int uint8_num = dim / 8;
    uint8_t* ts = new uint8_t[rows * uint8_num];
    memcpy(ts, data, rows * uint8_num * sizeof(uint8_t));
    auto ds = knowhere::GenDataSet(rows, dim, ts);
    ds->SetIsOwner(true);
    return ds;
}

inline knowhere::DataSetPtr
GenIdsDataSet(int rows, int nq, int64_t seed = 42) {
    std::mt19937 g(seed);
    int64_t* ids = new int64_t[rows];
    for (int i = 0; i < rows; ++i) ids[i] = i;
    std::shuffle(ids, ids + rows, g);
    auto ds = knowhere::GenIdsDataSet(nq, ids);
    ds->SetIsOwner(true);
    return ds;
}

inline knowhere::DataSetPtr
GenIdsDataSet(int rows, std::vector<int64_t>& ids) {
    auto ds = knowhere::GenIdsDataSet(rows, ids.data());
    ds->SetIsOwner(false);
    return ds;
}

inline float
GetKNNRecall(const knowhere::DataSet& ground_truth, const knowhere::DataSet& result) {
    REQUIRE(ground_truth.GetDim() >= result.GetDim());

    auto nq = result.GetRows();
    auto gt_k = ground_truth.GetDim();
    auto res_k = result.GetDim();
    auto gt_ids = ground_truth.GetIds();
    auto res_ids = result.GetIds();

    uint32_t matched_num = 0;
    for (auto i = 0; i < nq; ++i) {
        std::vector<int64_t> ids_0(gt_ids + i * gt_k, gt_ids + i * gt_k + res_k);
        std::vector<int64_t> ids_1(res_ids + i * res_k, res_ids + i * res_k + res_k);

        std::sort(ids_0.begin(), ids_0.end());
        std::sort(ids_1.begin(), ids_1.end());

        std::vector<int64_t> v(std::max(ids_0.size(), ids_1.size()));
        std::vector<int64_t>::iterator it;
        it = std::set_intersection(ids_0.begin(), ids_0.end(), ids_1.begin(), ids_1.end(), v.begin());
        v.resize(it - v.begin());
        matched_num += v.size();
    }
    return ((float)matched_num) / ((float)nq * res_k);
}

inline float
GetKNNRecall(const knowhere::DataSet& ground_truth, const std::vector<std::vector<int64_t>>& result) {
    auto nq = result.size();
    auto gt_k = ground_truth.GetDim();
    auto gt_ids = ground_truth.GetIds();

    uint32_t matched_num = 0;
    for (size_t i = 0; i < nq; ++i) {
        std::vector<int64_t> ids_0(gt_ids + i * gt_k, gt_ids + i * gt_k + gt_k);
        std::vector<int64_t> ids_1 = result[i];

        std::sort(ids_0.begin(), ids_0.end());
        std::sort(ids_1.begin(), ids_1.end());

        std::vector<int64_t> v(std::max(ids_0.size(), ids_1.size()));
        std::vector<int64_t>::iterator it;
        it = std::set_intersection(ids_0.begin(), ids_0.end(), ids_1.begin(), ids_1.end(), v.begin());
        v.resize(it - v.begin());
        matched_num += v.size();
    }
    return ((float)matched_num) / ((float)nq * gt_k);
}

//  Compare two ann-search results
//      "ground_truth" here is just used as a baseline value for comparison. It is not real groundtruth and the knn
//  results may be worse, we can call the compare results as "relative-recall".
//      when the k-th distance of gt is worse, define the recall as 1.0f
//      when the k-th distance of gt is better, define the recall as (intersection_count / size)
inline float
GetKNNRelativeRecall(const knowhere::DataSet& ground_truth, const knowhere::DataSet& result, bool dist_less_better) {
    REQUIRE(ground_truth.GetDim() >= result.GetDim());

    auto nq = result.GetRows();
    auto gt_k = ground_truth.GetDim();
    auto res_k = result.GetDim();
    auto gt_ids = ground_truth.GetIds();
    auto res_ids = result.GetIds();
    auto gt_dists = ground_truth.GetDistance();
    auto res_dists = result.GetDistance();

    double acc_recall = 0;
    for (auto i = 0; i < nq; ++i) {
        // Results may be insufficient, less than k
        int64_t valid_gt_count = 0;
        while (valid_gt_count < gt_k && gt_ids[i * gt_k + valid_gt_count] >= 0) {
            valid_gt_count++;
        }

        if (valid_gt_count == 0) {
            acc_recall += 1.0;
            continue;
        }

        bool gt_better = dist_less_better
                             ? gt_dists[i * gt_k + valid_gt_count - 1] < res_dists[i * res_k + valid_gt_count - 1]
                             : gt_dists[i * gt_k + valid_gt_count - 1] > res_dists[i * res_k + valid_gt_count - 1];

        if (gt_better) {
            std::vector<int64_t> ids_0(gt_ids + i * gt_k, gt_ids + i * gt_k + valid_gt_count);
            std::vector<int64_t> ids_1(res_ids + i * res_k, res_ids + i * res_k + valid_gt_count);

            std::sort(ids_0.begin(), ids_0.end());
            std::sort(ids_1.begin(), ids_1.end());

            std::vector<int64_t> v(std::max(ids_0.size(), ids_1.size()));
            std::vector<int64_t>::iterator it;
            it = std::set_intersection(ids_0.begin(), ids_0.end(), ids_1.begin(), ids_1.end(), v.begin());
            v.resize(it - v.begin());

            acc_recall += double(v.size()) / valid_gt_count;
        } else {
            acc_recall += 1.0;
        }
    }
    return acc_recall / nq;
}

inline float
GetRangeSearchRecall(const knowhere::DataSet& gt, const knowhere::DataSet& result) {
    uint32_t nq = result.GetRows();
    auto res_ids_p = result.GetIds();
    auto res_lims_p = result.GetLims();
    auto gt_ids_p = gt.GetIds();
    auto gt_lims_p = gt.GetLims();

    // check if both gt and result are empty
    if (gt_lims_p[nq] == 0 && res_lims_p[nq] == 0) {
        return 1;
    }

    uint32_t ninter = 0;
    for (uint32_t i = 0; i < nq; ++i) {
        std::set<int64_t> inter;
        std::set<int64_t> res_ids_set(res_ids_p + res_lims_p[i], res_ids_p + res_lims_p[i + 1]);
        std::set<int64_t> gt_ids_set(gt_ids_p + gt_lims_p[i], gt_ids_p + gt_lims_p[i + 1]);
        std::set_intersection(res_ids_set.begin(), res_ids_set.end(), gt_ids_set.begin(), gt_ids_set.end(),
                              std::inserter(inter, inter.begin()));
        ninter += inter.size();
    }

    float recall = ninter * 1.0f / gt_lims_p[nq];
    float precision = ninter * 1.0f / res_lims_p[nq];

    return (1 + precision) * recall / 2;
}

inline float
GetRelativeLoss(float gt_res, float res) {
    if (gt_res == 0.0 || std::abs(gt_res) < 0.000001) {
        return gt_res;
    }
    return std::abs((gt_res - res) / gt_res);
}

inline bool
CheckDistanceInScope(const knowhere::DataSet& result, int topk, float low_bound, float high_bound) {
    auto ids = result.GetIds();
    auto distances = result.GetDistance();
    auto rows = result.GetRows();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < topk; j++) {
            auto idx = i * topk + j;
            auto id = ids[idx];
            auto d = distances[idx];
            if (id != -1 && !(low_bound < d && d < high_bound)) {
                return false;
            }
        }
    }
    return true;
}

inline bool
CheckDistanceInScope(const knowhere::DataSet& result, float low_bound, float high_bound) {
    auto ids = result.GetIds();
    auto distances = result.GetDistance();
    auto lims = result.GetLims();
    auto rows = result.GetRows();
    for (int i = 0; i < rows; ++i) {
        for (size_t j = lims[i]; j < lims[i + 1]; j++) {
            auto id = ids[j];
            auto d = distances[j];
            if (id != -1 && !(low_bound < d && d < high_bound)) {
                return false;
            }
        }
    }
    return true;
}

inline std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>>
GenerateScalarInfo(size_t n) {
    std::vector<std::vector<uint32_t>> scalar_info;
    scalar_info.reserve(2);
    std::vector<uint32_t> scalar1;
    scalar1.reserve(n / 2);
    std::vector<uint32_t> scalar2;
    scalar2.reserve(n - n / 2);
    for (size_t i = 0; i < n; ++i) {
        if (i % 2 == 0) {
            scalar2.emplace_back(i);
        } else {
            scalar1.emplace_back(i);
        }
    }
    scalar_info.emplace_back(std::move(scalar1));
    scalar_info.emplace_back(std::move(scalar2));
    std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>> scalar_map;
    scalar_map[0] = std::move(scalar_info);
    return scalar_map;
}

inline std::vector<uint8_t>
GenerateBitsetByScalarInfoAndFirstTBits(const std::vector<uint32_t>& scalar, size_t n, size_t t) {
    assert(scalar.size() <= n);
    assert(t >= 0 && t <= n - scalar.size());
    std::vector<uint8_t> data((n + 8 - 1) / 8, 0);
    // set bits by scalar info
    for (size_t i = 0; i < scalar.size(); ++i) {
        data[scalar[i] >> 3] |= (0x1 << (scalar[i] & 0x7));
    }
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (count == t) {
            break;
        }
        // already set, skip
        if (data[i >> 3] & (0x1 << (i & 0x7))) {
            continue;
        }
        data[i >> 3] |= (0x1 << (i & 0x7));
        ++count;
    }
    return data;
}

// Return a n-bits bitset data with first t bits set to true
inline std::vector<uint8_t>
GenerateBitsetWithFirstTbitsSet(size_t n, size_t t) {
    assert(t >= 0 && t <= n);
    std::vector<uint8_t> data((n + 8 - 1) / 8, 0);
    for (size_t i = 0; i < t; ++i) {
        data[i >> 3] |= (0x1 << (i & 0x7));
    }
    return data;
}

// Return a n-bits bitset data with random t bits set to true
inline std::vector<uint8_t>
GenerateBitsetWithRandomTbitsSet(size_t n, size_t t) {
    assert(t >= 0 && t <= n);
    std::vector<bool> bits_shuffle(n, false);
    for (size_t i = 0; i < t; ++i) bits_shuffle[i] = true;
    std::mt19937 g(kSeed);
    std::shuffle(bits_shuffle.begin(), bits_shuffle.end(), g);
    std::vector<uint8_t> data((n + 8 - 1) / 8, 0);
    for (size_t i = 0; i < n; ++i) {
        if (bits_shuffle[i]) {
            data[i >> 3] |= (0x1 << (i & 0x7));
        }
    }
    return data;
}

// Randomly generate n (distances, id) pairs
inline std::vector<std::pair<float, size_t>>
GenerateRandomDistanceIdPair(size_t n) {
    std::mt19937 rng(kSeed);
    std::uniform_real_distribution<> distrib(std::numeric_limits<float>().min(), std::numeric_limits<float>().max());
    std::vector<std::pair<float, size_t>> res;
    res.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        res.emplace_back(distrib(rng), i);
    }
    return res;
}

inline auto
GenTestVersionList() {
    return GENERATE(as<int32_t>{}, knowhere::Version::GetCurrentVersion().VersionNumber());
}

inline knowhere::DataSetPtr
GenSparseDataSet(const std::vector<std::map<int32_t, float>>& data, int32_t cols) {
    int32_t rows = data.size();
    auto tensor = std::make_unique<knowhere::sparse::SparseRow<float>[]>(rows);

    for (int32_t i = 0; i < rows; ++i) {
        if (data[i].size() == 0) {
            continue;
        }
        knowhere::sparse::SparseRow<float> row(data[i].size());
        size_t j = 0;
        for (auto& [idx, val] : data[i]) {
            row.set_at(j++, idx, val);
        }
        tensor[i] = std::move(row);
    }

    auto ds = knowhere::GenDataSet(rows, cols, tensor.release());
    ds->SetIsOwner(true);
    ds->SetIsSparse(true);
    return ds;
}

// Generate a sparse dataset with given sparsity.
inline knowhere::DataSetPtr
GenSparseDataSet(int32_t rows, int32_t cols, float sparsity, int seed = 42) {
    int32_t num_elements = static_cast<int32_t>(rows * cols * (1.0f - sparsity));

    std::mt19937 rng(seed);
    auto real_distrib = std::uniform_real_distribution<float>(0, 1);
    auto row_distrib = std::uniform_int_distribution<int32_t>(0, rows - 1);
    auto col_distrib = std::uniform_int_distribution<int32_t>(0, cols - 1);

    std::vector<std::map<int32_t, float>> data(rows);

    for (int32_t i = 0; i < num_elements; ++i) {
        auto row = row_distrib(rng);
        while (data[row].size() == (size_t)cols) {
            row = row_distrib(rng);
        }
        auto col = col_distrib(rng);
        while (data[row].find(col) != data[row].end()) {
            col = col_distrib(rng);
        }
        auto val = real_distrib(rng);
        data[row][col] = val;
    }

    return GenSparseDataSet(data, cols);
}

// Generate a sparse dataset with given sparsity and max value.
inline knowhere::DataSetPtr
GenSparseDataSetWithMaxVal(int32_t rows, int32_t cols, float sparsity, float max_val, bool use_bm25 = false,
                           int seed = 42) {
    int32_t num_elements = static_cast<int32_t>(rows * cols * (1.0f - sparsity));

    std::mt19937 rng(seed);
    auto real_distrib = std::uniform_real_distribution<float>(0, max_val);
    auto row_distrib = std::uniform_int_distribution<int32_t>(0, rows - 1);
    auto col_distrib = std::uniform_int_distribution<int32_t>(0, cols - 1);

    std::vector<std::map<int32_t, float>> data(rows);

    for (int32_t i = 0; i < num_elements; ++i) {
        auto row = row_distrib(rng);
        while (data[row].size() == (size_t)cols) {
            row = row_distrib(rng);
        }
        auto col = col_distrib(rng);
        while (data[row].find(col) != data[row].end()) {
            col = col_distrib(rng);
        }
        auto val = use_bm25 ? static_cast<float>(static_cast<int32_t>(real_distrib(rng))) : real_distrib(rng);
        data[row][col] = val;
    }

    return GenSparseDataSet(data, cols);
}

// a timer
struct StopWatch {
    using timepoint_t = std::chrono::time_point<std::chrono::steady_clock>;

    timepoint_t Start;

    //
    StopWatch() {
        Start = std::chrono::steady_clock::now();
    }

    //
    double
    elapsed() const {
        const auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = now - Start;
        return elapsed.count();
    }
};
