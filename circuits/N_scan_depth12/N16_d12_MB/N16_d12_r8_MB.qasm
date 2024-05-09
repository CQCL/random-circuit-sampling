OPENQASM 2.0;
include "hqslib1.inc";

qreg q[16];
creg c[16];
U1q(0.226347682535607*pi,1.016009078170217*pi) q[0];
U1q(0.281949056256241*pi,1.9364011241725498*pi) q[1];
U1q(0.16312231651188*pi,0.316037714024515*pi) q[2];
U1q(1.93452537328335*pi,1.5240821308569863*pi) q[3];
U1q(1.67418316060232*pi,0.5815792894330526*pi) q[4];
U1q(0.726416438908961*pi,1.44806910628007*pi) q[5];
U1q(0.554635176680257*pi,0.567032509167291*pi) q[6];
U1q(0.388550935284049*pi,1.478965418907404*pi) q[7];
U1q(0.65391158432882*pi,1.33150410626181*pi) q[8];
U1q(0.613808967341493*pi,1.22034591370584*pi) q[9];
U1q(0.607534843927843*pi,0.401309645879267*pi) q[10];
U1q(1.85813682286744*pi,0.5217557112802276*pi) q[11];
U1q(1.38887217052189*pi,1.3767105719276018*pi) q[12];
U1q(0.0202623561153692*pi,0.559504658456838*pi) q[13];
U1q(1.26695911862595*pi,0.6516661354103478*pi) q[14];
U1q(1.61999809558797*pi,1.5061564784301669*pi) q[15];
RZZ(0.5*pi) q[14],q[0];
RZZ(0.5*pi) q[12],q[1];
RZZ(0.5*pi) q[8],q[2];
RZZ(0.5*pi) q[3],q[10];
RZZ(0.5*pi) q[4],q[9];
RZZ(0.5*pi) q[6],q[5];
RZZ(0.5*pi) q[11],q[7];
RZZ(0.5*pi) q[13],q[15];
U1q(0.610961325662961*pi,0.20410929831858993*pi) q[0];
U1q(0.686246196225018*pi,0.2743440300686*pi) q[1];
U1q(0.587423687454771*pi,0.17657818296363015*pi) q[2];
U1q(0.772279587842122*pi,0.6519051631970165*pi) q[3];
U1q(0.722477505055692*pi,1.8605067069483625*pi) q[4];
U1q(0.595948275715796*pi,0.9823258359746001*pi) q[5];
U1q(0.56215990616231*pi,1.6904120494728998*pi) q[6];
U1q(0.541229125830111*pi,0.061362524603900015*pi) q[7];
U1q(0.501361914375785*pi,1.7657036651536102*pi) q[8];
U1q(0.550974910793059*pi,1.046229785640851*pi) q[9];
U1q(0.922745254737971*pi,0.78836610297327*pi) q[10];
U1q(0.402775985267434*pi,0.11320813944645725*pi) q[11];
U1q(0.808970120093858*pi,0.9237345675709219*pi) q[12];
U1q(0.193307900449394*pi,0.78065495112068*pi) q[13];
U1q(0.59385782772011*pi,1.790698509573538*pi) q[14];
U1q(0.411882743118369*pi,1.9212705288045973*pi) q[15];
RZZ(0.5*pi) q[4],q[0];
RZZ(0.5*pi) q[10],q[1];
RZZ(0.5*pi) q[9],q[2];
RZZ(0.5*pi) q[3],q[5];
RZZ(0.5*pi) q[6],q[7];
RZZ(0.5*pi) q[15],q[8];
RZZ(0.5*pi) q[14],q[11];
RZZ(0.5*pi) q[12],q[13];
U1q(0.104162674264342*pi,0.7854006677972101*pi) q[0];
U1q(0.908188609727268*pi,0.4321932065174401*pi) q[1];
U1q(0.52936083635692*pi,1.64561610495352*pi) q[2];
U1q(0.584627162233391*pi,0.8466270730056964*pi) q[3];
U1q(0.130312453380953*pi,1.947603413259162*pi) q[4];
U1q(0.749477213431014*pi,0.3253999725846799*pi) q[5];
U1q(0.360531964199648*pi,1.1241134327723001*pi) q[6];
U1q(0.852188698224491*pi,1.3508342662152497*pi) q[7];
U1q(0.70009649624905*pi,1.4020523031736003*pi) q[8];
U1q(0.112937072181035*pi,0.6884738196677498*pi) q[9];
U1q(0.667369177271613*pi,0.6857339431250304*pi) q[10];
U1q(0.33265502560699*pi,1.383845838983377*pi) q[11];
U1q(0.849076702074397*pi,0.0035596015726220287*pi) q[12];
U1q(0.384576031583547*pi,0.9234788416105699*pi) q[13];
U1q(0.0960007050956613*pi,1.9786193820522673*pi) q[14];
U1q(0.386221662839441*pi,1.0742970740617572*pi) q[15];
RZZ(0.5*pi) q[5],q[0];
RZZ(0.5*pi) q[1],q[7];
RZZ(0.5*pi) q[14],q[2];
RZZ(0.5*pi) q[3],q[8];
RZZ(0.5*pi) q[4],q[15];
RZZ(0.5*pi) q[6],q[13];
RZZ(0.5*pi) q[9],q[12];
RZZ(0.5*pi) q[10],q[11];
U1q(0.234931961840882*pi,1.5190177712093602*pi) q[0];
U1q(0.833239619403661*pi,1.60781417438558*pi) q[1];
U1q(0.794570182496148*pi,1.2123751182709999*pi) q[2];
U1q(0.5898160303871*pi,1.3794899625971864*pi) q[3];
U1q(0.477580495565248*pi,1.261249928290682*pi) q[4];
U1q(0.343598968105853*pi,0.44676642871664995*pi) q[5];
U1q(0.368461004538774*pi,1.39627896095541*pi) q[6];
U1q(0.697204839302688*pi,0.15656259367565983*pi) q[7];
U1q(0.50074541096567*pi,0.8004795305690999*pi) q[8];
U1q(0.248048801957593*pi,1.6199839856560203*pi) q[9];
U1q(0.421678260090021*pi,0.03589241547586042*pi) q[10];
U1q(0.31889277166875*pi,0.26360421492272756*pi) q[11];
U1q(0.299397162971203*pi,1.452855366335851*pi) q[12];
U1q(0.804688186761554*pi,1.7290429543321002*pi) q[13];
U1q(0.36255708838384*pi,1.4483262516067876*pi) q[14];
U1q(0.690749148407177*pi,1.0539477524695666*pi) q[15];
RZZ(0.5*pi) q[13],q[0];
RZZ(0.5*pi) q[5],q[1];
RZZ(0.5*pi) q[15],q[2];
RZZ(0.5*pi) q[12],q[3];
RZZ(0.5*pi) q[4],q[6];
RZZ(0.5*pi) q[8],q[7];
RZZ(0.5*pi) q[9],q[11];
RZZ(0.5*pi) q[10],q[14];
U1q(0.34928212215743*pi,1.2897524989632707*pi) q[0];
U1q(0.641088254138222*pi,0.26916397579265006*pi) q[1];
U1q(0.34363927449183*pi,0.15278725617196987*pi) q[2];
U1q(0.108376970064145*pi,1.201895076165055*pi) q[3];
U1q(0.513319824985497*pi,0.2504443268493528*pi) q[4];
U1q(0.320122100313974*pi,1.1008274994668206*pi) q[5];
U1q(0.902845873780164*pi,1.4270884400017003*pi) q[6];
U1q(0.644814977951708*pi,1.6415807939707001*pi) q[7];
U1q(0.467560282169182*pi,1.2018167636197408*pi) q[8];
U1q(0.57935143035623*pi,0.1841191190118101*pi) q[9];
U1q(0.632126720285701*pi,0.4853879038497997*pi) q[10];
U1q(0.516105714246326*pi,1.0046601826031285*pi) q[11];
U1q(0.505359806126356*pi,0.8731701221319614*pi) q[12];
U1q(0.589251285697906*pi,1.4403555597009596*pi) q[13];
U1q(0.249167115934826*pi,0.42427520567644805*pi) q[14];
U1q(0.655312405202291*pi,0.14843265779446746*pi) q[15];
RZZ(0.5*pi) q[6],q[0];
RZZ(0.5*pi) q[4],q[1];
RZZ(0.5*pi) q[3],q[2];
RZZ(0.5*pi) q[5],q[15];
RZZ(0.5*pi) q[13],q[7];
RZZ(0.5*pi) q[11],q[8];
RZZ(0.5*pi) q[9],q[14];
RZZ(0.5*pi) q[12],q[10];
U1q(0.692215817979048*pi,0.16178119879869968*pi) q[0];
U1q(0.602430055697298*pi,1.6909313611144503*pi) q[1];
U1q(0.660078121233318*pi,1.2533190181080993*pi) q[2];
U1q(0.280922572185335*pi,0.6029802225563863*pi) q[3];
U1q(0.678221452110274*pi,1.582100418523254*pi) q[4];
U1q(0.243312503653692*pi,1.8123851770361004*pi) q[5];
U1q(0.539685045670546*pi,1.7062279665230005*pi) q[6];
U1q(0.497538069254827*pi,0.05811276510280017*pi) q[7];
U1q(0.56847907723582*pi,1.2498525721592006*pi) q[8];
U1q(0.592540451178185*pi,0.5897722721106806*pi) q[9];
U1q(0.303631766928009*pi,1.1096455622146006*pi) q[10];
U1q(0.521263223017262*pi,1.5492298698591274*pi) q[11];
U1q(0.346216524961125*pi,0.6911463727621019*pi) q[12];
U1q(0.171253836023622*pi,1.7562432926863991*pi) q[13];
U1q(0.245163511614199*pi,1.7444945200900488*pi) q[14];
U1q(0.403319951550854*pi,1.0070278888777686*pi) q[15];
RZZ(0.5*pi) q[12],q[0];
RZZ(0.5*pi) q[14],q[1];
RZZ(0.5*pi) q[10],q[2];
RZZ(0.5*pi) q[3],q[11];
RZZ(0.5*pi) q[4],q[7];
RZZ(0.5*pi) q[13],q[5];
RZZ(0.5*pi) q[6],q[8];
RZZ(0.5*pi) q[9],q[15];
U1q(0.530333609445771*pi,1.1359055261421993*pi) q[0];
U1q(0.361909215003105*pi,1.2524760765360003*pi) q[1];
U1q(0.315853142509709*pi,0.04433698362229954*pi) q[2];
U1q(0.673619696695654*pi,1.5862997802855858*pi) q[3];
U1q(0.767331263865402*pi,0.0589981592943527*pi) q[4];
U1q(0.615924633213868*pi,0.7444879525586003*pi) q[5];
U1q(0.412903747169518*pi,1.2829068832922985*pi) q[6];
U1q(0.431387928376577*pi,1.6794684838581997*pi) q[7];
U1q(0.663687269989978*pi,1.0355930489374003*pi) q[8];
U1q(0.62428659650548*pi,1.3459961842678005*pi) q[9];
U1q(0.337637062523048*pi,0.24356879234399997*pi) q[10];
U1q(0.427165788525245*pi,1.6085457498334286*pi) q[11];
U1q(0.12869389798599*pi,0.023704244842601696*pi) q[12];
U1q(0.704506885199538*pi,0.8301124430380007*pi) q[13];
U1q(0.197674332890119*pi,0.8250426055236488*pi) q[14];
U1q(0.518240144948415*pi,1.8393217088141682*pi) q[15];
rz(3.6334215259500997*pi) q[0];
rz(1.5274469259759398*pi) q[1];
rz(0.6048267378751007*pi) q[2];
rz(0.4832402120040147*pi) q[3];
rz(1.657574054089448*pi) q[4];
rz(1.3443127628382996*pi) q[5];
rz(0.3939023091466005*pi) q[6];
rz(2.8031926692785003*pi) q[7];
rz(1.6636994196475996*pi) q[8];
rz(2.6673464382195*pi) q[9];
rz(0.6406920148260014*pi) q[10];
rz(1.7940842519457725*pi) q[11];
rz(2.1481663756569986*pi) q[12];
rz(3.9233690666077994*pi) q[13];
rz(1.4112419975013513*pi) q[14];
rz(1.1525487574496331*pi) q[15];
U1q(0.530333609445771*pi,1.769327052092339*pi) q[0];
U1q(0.361909215003105*pi,1.779923002511943*pi) q[1];
U1q(0.315853142509709*pi,1.649163721497381*pi) q[2];
U1q(1.67361969669565*pi,1.06953999228953*pi) q[3];
U1q(0.767331263865402*pi,0.716572213383877*pi) q[4];
U1q(1.61592463321387*pi,1.08880071539691*pi) q[5];
U1q(0.412903747169518*pi,0.6768091924389199*pi) q[6];
U1q(1.43138792837658*pi,1.482661153136727*pi) q[7];
U1q(0.663687269989978*pi,1.699292468584946*pi) q[8];
U1q(1.62428659650548*pi,1.01334262248727*pi) q[9];
U1q(3.337637062523048*pi,1.884260807169973*pi) q[10];
U1q(1.42716578852525*pi,0.402630001779165*pi) q[11];
U1q(0.12869389798599*pi,1.1718706204995462*pi) q[12];
U1q(1.70450688519954*pi,1.753481509645779*pi) q[13];
U1q(0.197674332890119*pi,1.2362846030250552*pi) q[14];
U1q(0.518240144948415*pi,1.9918704662637912*pi) q[15];
RZZ(0.5*pi) q[12],q[0];
RZZ(0.5*pi) q[14],q[1];
RZZ(0.5*pi) q[10],q[2];
RZZ(0.5*pi) q[3],q[11];
RZZ(0.5*pi) q[4],q[7];
RZZ(0.5*pi) q[13],q[5];
RZZ(0.5*pi) q[6],q[8];
RZZ(0.5*pi) q[9],q[15];
U1q(1.69221581797905*pi,1.79520272474879*pi) q[0];
U1q(0.602430055697298*pi,1.21837828709039*pi) q[1];
U1q(0.660078121233318*pi,0.858145755983177*pi) q[2];
U1q(1.28092257218533*pi,1.052859550018702*pi) q[3];
U1q(0.678221452110274*pi,0.23967447261277997*pi) q[4];
U1q(1.24331250365369*pi,1.0209034909193604*pi) q[5];
U1q(3.539685045670546*pi,0.10013027566960009*pi) q[6];
U1q(3.5024619307451728*pi,0.10401687189217368*pi) q[7];
U1q(3.56847907723582*pi,0.91355199180677*pi) q[8];
U1q(1.59254045117819*pi,1.769566534644363*pi) q[9];
U1q(1.30363176692801*pi,1.0181840372994255*pi) q[10];
U1q(3.478736776982738*pi,0.4619458817534703*pi) q[11];
U1q(1.34621652496113*pi,0.8393127484190601*pi) q[12];
U1q(3.828746163976378*pi,0.827350659997399*pi) q[13];
U1q(1.2451635116142*pi,1.1557365175914232*pi) q[14];
U1q(0.403319951550854*pi,1.15957664632738*pi) q[15];
RZZ(0.5*pi) q[6],q[0];
RZZ(0.5*pi) q[4],q[1];
RZZ(0.5*pi) q[3],q[2];
RZZ(0.5*pi) q[5],q[15];
RZZ(0.5*pi) q[13],q[7];
RZZ(0.5*pi) q[11],q[8];
RZZ(0.5*pi) q[9],q[14];
RZZ(0.5*pi) q[12],q[10];
U1q(3.34928212215743*pi,1.6672314245842106*pi) q[0];
U1q(0.641088254138222*pi,0.7966109017685898*pi) q[1];
U1q(0.34363927449183*pi,0.7576139940470199*pi) q[2];
U1q(1.10837697006414*pi,0.6517744036273937*pi) q[3];
U1q(0.513319824985497*pi,1.9080183809388398*pi) q[4];
U1q(0.320122100313974*pi,1.3093458133500537*pi) q[5];
U1q(3.902845873780164*pi,1.3792698021908834*pi) q[6];
U1q(1.64481497795171*pi,0.5205488430242098*pi) q[7];
U1q(3.532439717830818*pi,0.9615878003462017*pi) q[8];
U1q(3.579351430356231*pi,1.3639133815454931*pi) q[9];
U1q(0.632126720285701*pi,0.3939263789346885*pi) q[10];
U1q(3.516105714246326*pi,1.0065155690094816*pi) q[11];
U1q(3.5053598061263562*pi,0.6572889990492037*pi) q[12];
U1q(3.410748714302093*pi,0.1432383929828287*pi) q[13];
U1q(1.24916711593483*pi,0.47595583200503166*pi) q[14];
U1q(0.655312405202291*pi,1.300981415244082*pi) q[15];
RZZ(0.5*pi) q[13],q[0];
RZZ(0.5*pi) q[5],q[1];
RZZ(0.5*pi) q[15],q[2];
RZZ(0.5*pi) q[12],q[3];
RZZ(0.5*pi) q[4],q[6];
RZZ(0.5*pi) q[8],q[7];
RZZ(0.5*pi) q[9],q[11];
RZZ(0.5*pi) q[10],q[14];
U1q(0.234931961840882*pi,1.896496696830301*pi) q[0];
U1q(0.833239619403661*pi,1.1352611003615198*pi) q[1];
U1q(0.794570182496148*pi,0.8172018561460499*pi) q[2];
U1q(3.589816030387101*pi,0.47417951719526297*pi) q[3];
U1q(0.477580495565248*pi,1.9188239823801698*pi) q[4];
U1q(0.343598968105853*pi,1.6552847425998847*pi) q[5];
U1q(1.36846100453877*pi,0.3484603231446233*pi) q[6];
U1q(1.69720483930269*pi,1.0355306427291397*pi) q[7];
U1q(3.499254589034329*pi,1.3629250333968517*pi) q[8];
U1q(1.24804880195759*pi,1.9280485149012856*pi) q[9];
U1q(0.421678260090021*pi,0.9444308905607324*pi) q[10];
U1q(0.31889277166875*pi,0.26545960132910107*pi) q[11];
U1q(0.299397162971203*pi,1.2369742432530937*pi) q[12];
U1q(3.195311813238447*pi,1.8545509983516886*pi) q[13];
U1q(0.36255708838384*pi,0.5000068779353688*pi) q[14];
U1q(1.69074914840718*pi,1.20649650991913*pi) q[15];
RZZ(0.5*pi) q[5],q[0];
RZZ(0.5*pi) q[1],q[7];
RZZ(0.5*pi) q[14],q[2];
RZZ(0.5*pi) q[3],q[8];
RZZ(0.5*pi) q[4],q[15];
RZZ(0.5*pi) q[6],q[13];
RZZ(0.5*pi) q[9],q[12];
RZZ(0.5*pi) q[10],q[11];
U1q(0.104162674264342*pi,0.1628795934181504*pi) q[0];
U1q(1.90818860972727*pi,1.9596401324933908*pi) q[1];
U1q(0.52936083635692*pi,0.25044284282857987*pi) q[2];
U1q(1.58462716223339*pi,1.9413166276037606*pi) q[3];
U1q(0.130312453380953*pi,0.6051774673486401*pi) q[4];
U1q(1.74947721343101*pi,0.5339182864679146*pi) q[5];
U1q(1.36053196419965*pi,1.6206258513277323*pi) q[6];
U1q(1.85218869822449*pi,1.841258970189541*pi) q[7];
U1q(1.70009649624905*pi,1.761352260792357*pi) q[8];
U1q(0.112937072181035*pi,1.996538348913015*pi) q[9];
U1q(1.66736917727161*pi,0.5942724182099024*pi) q[10];
U1q(0.33265502560699*pi,0.3857012253897718*pi) q[11];
U1q(1.8490767020744*pi,0.7876784784898536*pi) q[12];
U1q(3.615423968416453*pi,1.6601151110732189*pi) q[13];
U1q(1.09600070509566*pi,0.030300008380848453*pi) q[14];
U1q(1.38622166283944*pi,1.186147188326903*pi) q[15];
RZZ(0.5*pi) q[4],q[0];
RZZ(0.5*pi) q[10],q[1];
RZZ(0.5*pi) q[9],q[2];
RZZ(0.5*pi) q[3],q[5];
RZZ(0.5*pi) q[6],q[7];
RZZ(0.5*pi) q[15],q[8];
RZZ(0.5*pi) q[14],q[11];
RZZ(0.5*pi) q[12],q[13];
U1q(3.610961325662961*pi,0.58158822393953*pi) q[0];
U1q(1.68624619622502*pi,1.1174893089422309*pi) q[1];
U1q(0.587423687454771*pi,1.7814049208386802*pi) q[2];
U1q(1.77227958784212*pi,1.1360385374124449*pi) q[3];
U1q(0.722477505055692*pi,1.5180807610377993*pi) q[4];
U1q(3.404051724284204*pi,1.8769924230779997*pi) q[5];
U1q(1.56215990616231*pi,1.186924468028332*pi) q[6];
U1q(0.541229125830111*pi,1.551787228578191*pi) q[7];
U1q(0.501361914375785*pi,1.125003622772367*pi) q[8];
U1q(0.550974910793059*pi,1.3542943148861148*pi) q[9];
U1q(3.077254745262029*pi,0.49164025836165504*pi) q[10];
U1q(3.402775985267434*pi,1.1150635258528512*pi) q[11];
U1q(3.808970120093859*pi,0.867503512491556*pi) q[12];
U1q(1.19330790044939*pi,1.802939001563114*pi) q[13];
U1q(3.40614217227989*pi,1.2182208808595725*pi) q[14];
U1q(1.41188274311837*pi,1.0331206430697426*pi) q[15];
RZZ(0.5*pi) q[14],q[0];
RZZ(0.5*pi) q[12],q[1];
RZZ(0.5*pi) q[8],q[2];
RZZ(0.5*pi) q[3],q[10];
RZZ(0.5*pi) q[4],q[9];
RZZ(0.5*pi) q[6],q[5];
RZZ(0.5*pi) q[11],q[7];
RZZ(0.5*pi) q[13],q[15];
U1q(1.22634768253561*pi,0.7696884440879046*pi) q[0];
U1q(0.281949056256241*pi,1.7795464030461403*pi) q[1];
U1q(0.16312231651188*pi,0.9208644518995701*pi) q[2];
U1q(0.93452537328335*pi,0.00821550507241664*pi) q[3];
U1q(0.674183160602324*pi,1.2391533435224993*pi) q[4];
U1q(1.72641643890896*pi,1.4112491527725295*pi) q[5];
U1q(1.55463517668026*pi,1.3103040083339028*pi) q[6];
U1q(0.388550935284049*pi,0.9693901228816904*pi) q[7];
U1q(0.65391158432882*pi,1.690804063880567*pi) q[8];
U1q(0.613808967341493*pi,0.5284104429511061*pi) q[9];
U1q(1.60753484392784*pi,1.8786967154556642*pi) q[10];
U1q(1.85813682286744*pi,0.7065159540190855*pi) q[11];
U1q(0.388872170521889*pi,0.3204795168482466*pi) q[12];
U1q(0.0202623561153692*pi,1.581788708899274*pi) q[13];
U1q(3.2669591186259472*pi,0.3572532550227612*pi) q[14];
U1q(1.61999809558797*pi,1.4482346934441708*pi) q[15];
rz(3.2303115559120954*pi) q[0];
rz(2.2204535969538597*pi) q[1];
rz(1.0791355481004299*pi) q[2];
rz(1.9917844949275834*pi) q[3];
rz(0.7608466564775007*pi) q[4];
rz(0.5887508472274705*pi) q[5];
rz(2.689695991666097*pi) q[6];
rz(3.0306098771183096*pi) q[7];
rz(2.309195936119433*pi) q[8];
rz(1.471589557048894*pi) q[9];
rz(0.12130328454433581*pi) q[10];
rz(1.2934840459809145*pi) q[11];
rz(3.6795204831517534*pi) q[12];
rz(2.418211291100726*pi) q[13];
rz(3.642746744977239*pi) q[14];
rz(0.5517653065558292*pi) q[15];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];
measure q[10] -> c[10];
measure q[11] -> c[11];
measure q[12] -> c[12];
measure q[13] -> c[13];
measure q[14] -> c[14];
measure q[15] -> c[15];