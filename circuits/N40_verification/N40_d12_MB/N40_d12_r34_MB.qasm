OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.0635961186697839*pi,0.623879878494835*pi) q[0];
U1q(1.67357683947169*pi,0.22337828279818142*pi) q[1];
U1q(3.80276597386224*pi,0.845666926457646*pi) q[2];
U1q(1.51448975306689*pi,1.7664656799563703*pi) q[3];
U1q(0.124918459652875*pi,1.827108457709679*pi) q[4];
U1q(1.04947923171976*pi,1.9808172198734957*pi) q[5];
U1q(0.487808319006538*pi,0.6404597445970801*pi) q[6];
U1q(1.53631820852483*pi,1.5111354980364569*pi) q[7];
U1q(1.38775428413157*pi,0.5020901958555035*pi) q[8];
U1q(1.39097982907642*pi,1.2606773155370081*pi) q[9];
U1q(1.72653830137214*pi,1.7336237578094558*pi) q[10];
U1q(3.524966320028316*pi,0.7388493068815124*pi) q[11];
U1q(1.75678786056527*pi,1.2570490102891094*pi) q[12];
U1q(1.5523950488547*pi,1.1164772346325689*pi) q[13];
U1q(3.280488773106326*pi,0.8821958189161825*pi) q[14];
U1q(3.0912925810672762*pi,1.3440079840988122*pi) q[15];
U1q(3.776100870162241*pi,0.5728616581767612*pi) q[16];
U1q(0.453757966453255*pi,1.829313989012746*pi) q[17];
U1q(0.552985125170573*pi,1.489019953086048*pi) q[18];
U1q(0.492444730307162*pi,0.283215839327149*pi) q[19];
U1q(0.495566025927019*pi,0.192708074299578*pi) q[20];
U1q(0.467167837466286*pi,1.629784396219513*pi) q[21];
U1q(0.792567110097512*pi,0.177205072175895*pi) q[22];
U1q(1.36532839301504*pi,0.6922803837998848*pi) q[23];
U1q(1.35514292641604*pi,0.9461412803296502*pi) q[24];
U1q(1.60904669481523*pi,0.26289240397824415*pi) q[25];
U1q(3.159988153375666*pi,1.0824219682783047*pi) q[26];
U1q(0.819775935735513*pi,0.536687063379391*pi) q[27];
U1q(1.25858545856217*pi,1.6263888152169081*pi) q[28];
U1q(0.713784365367372*pi,0.0144826057271727*pi) q[29];
U1q(0.408786441112893*pi,1.3932616691947168*pi) q[30];
U1q(0.65894948392087*pi,1.9577599399900034*pi) q[31];
U1q(0.708731682734186*pi,0.0706195195565343*pi) q[32];
U1q(3.621712080415341*pi,0.8650401775497835*pi) q[33];
U1q(3.594360204891*pi,0.7397091314219707*pi) q[34];
U1q(0.694856781051437*pi,0.772837372924187*pi) q[35];
U1q(0.61475059010873*pi,1.543366505036706*pi) q[36];
U1q(0.626069605237655*pi,1.600011259240505*pi) q[37];
U1q(0.279100531754125*pi,0.78522482000338*pi) q[38];
U1q(1.44824200384787*pi,1.4507521113449213*pi) q[39];
RZZ(0.5*pi) q[2],q[0];
RZZ(0.5*pi) q[1],q[35];
RZZ(0.5*pi) q[3],q[19];
RZZ(0.5*pi) q[4],q[27];
RZZ(0.5*pi) q[29],q[5];
RZZ(0.5*pi) q[30],q[6];
RZZ(0.5*pi) q[7],q[12];
RZZ(0.5*pi) q[33],q[8];
RZZ(0.5*pi) q[9],q[28];
RZZ(0.5*pi) q[10],q[34];
RZZ(0.5*pi) q[11],q[20];
RZZ(0.5*pi) q[13],q[16];
RZZ(0.5*pi) q[14],q[26];
RZZ(0.5*pi) q[15],q[18];
RZZ(0.5*pi) q[17],q[36];
RZZ(0.5*pi) q[37],q[21];
RZZ(0.5*pi) q[22],q[39];
RZZ(0.5*pi) q[23],q[31];
RZZ(0.5*pi) q[32],q[24];
RZZ(0.5*pi) q[25],q[38];
U1q(0.545818222469418*pi,0.2156781699563901*pi) q[0];
U1q(0.485903913483886*pi,1.7941529304510118*pi) q[1];
U1q(0.230180086266415*pi,0.886725513418426*pi) q[2];
U1q(0.757273387052579*pi,1.5643127855728602*pi) q[3];
U1q(0.272541853333136*pi,0.9524992531214602*pi) q[4];
U1q(0.263685441619673*pi,1.5452250795221456*pi) q[5];
U1q(0.586041343145436*pi,0.6150846841106201*pi) q[6];
U1q(0.763189444957732*pi,0.3338732485416669*pi) q[7];
U1q(0.66093585699897*pi,0.6560303448563136*pi) q[8];
U1q(0.512327423470307*pi,0.04532360653888823*pi) q[9];
U1q(0.459077749537214*pi,1.6375548608303556*pi) q[10];
U1q(0.900124985408001*pi,0.11463106749604035*pi) q[11];
U1q(0.356623034753223*pi,1.4314747138258892*pi) q[12];
U1q(0.627368492156322*pi,1.527204355959849*pi) q[13];
U1q(0.464484093268722*pi,1.5931378706729928*pi) q[14];
U1q(0.269809171166897*pi,0.2815524829836722*pi) q[15];
U1q(0.21270511153917*pi,0.028113372981161344*pi) q[16];
U1q(0.303547787600373*pi,0.41308687856886017*pi) q[17];
U1q(0.319682833833*pi,0.15002798047860022*pi) q[18];
U1q(0.29636030008226*pi,1.80101216365358*pi) q[19];
U1q(0.873242543989661*pi,0.3047836454778501*pi) q[20];
U1q(0.447391754004287*pi,0.6810538875783498*pi) q[21];
U1q(0.127556255689*pi,0.54557919135012*pi) q[22];
U1q(0.398205990557796*pi,0.1662313304216647*pi) q[23];
U1q(0.216645120159448*pi,1.8566995605050702*pi) q[24];
U1q(0.611490238090376*pi,0.3997459182178942*pi) q[25];
U1q(0.428288370378337*pi,0.6016136939978547*pi) q[26];
U1q(0.531060715434528*pi,1.124778792361302*pi) q[27];
U1q(0.825239720217543*pi,1.8595075065014788*pi) q[28];
U1q(0.302698220358478*pi,1.7167305339333199*pi) q[29];
U1q(0.772345717342069*pi,1.39291050308513*pi) q[30];
U1q(0.195534039536818*pi,1.7769057484450599*pi) q[31];
U1q(0.55363204456512*pi,1.6200959186628698*pi) q[32];
U1q(0.303327429070341*pi,1.7955918745780934*pi) q[33];
U1q(0.537693851015911*pi,0.22467211010548072*pi) q[34];
U1q(0.164593006561663*pi,0.15300519296672999*pi) q[35];
U1q(0.520108789668083*pi,1.70847544451457*pi) q[36];
U1q(0.927853474868702*pi,1.74535375556641*pi) q[37];
U1q(0.260191697535144*pi,1.2349968845881998*pi) q[38];
U1q(0.387992120290644*pi,1.3708070472970513*pi) q[39];
RZZ(0.5*pi) q[0],q[6];
RZZ(0.5*pi) q[13],q[1];
RZZ(0.5*pi) q[2],q[11];
RZZ(0.5*pi) q[3],q[35];
RZZ(0.5*pi) q[4],q[12];
RZZ(0.5*pi) q[5],q[28];
RZZ(0.5*pi) q[32],q[7];
RZZ(0.5*pi) q[8],q[31];
RZZ(0.5*pi) q[9],q[38];
RZZ(0.5*pi) q[10],q[18];
RZZ(0.5*pi) q[33],q[14];
RZZ(0.5*pi) q[15],q[19];
RZZ(0.5*pi) q[23],q[16];
RZZ(0.5*pi) q[17],q[22];
RZZ(0.5*pi) q[27],q[20];
RZZ(0.5*pi) q[21],q[30];
RZZ(0.5*pi) q[24],q[26];
RZZ(0.5*pi) q[34],q[25];
RZZ(0.5*pi) q[37],q[29];
RZZ(0.5*pi) q[39],q[36];
U1q(0.418166738192565*pi,1.7560667851570004*pi) q[0];
U1q(0.379912740237965*pi,0.3847279957496217*pi) q[1];
U1q(0.259099852788761*pi,1.7773990061077365*pi) q[2];
U1q(0.451170510175043*pi,1.703059914988721*pi) q[3];
U1q(0.401682885840488*pi,0.20617505475801012*pi) q[4];
U1q(0.177218997336359*pi,1.5514637046161353*pi) q[5];
U1q(0.737402852534316*pi,0.19086994535480972*pi) q[6];
U1q(0.85650367923329*pi,1.5445345944847868*pi) q[7];
U1q(0.225450658813872*pi,1.755249550217103*pi) q[8];
U1q(0.153120520834518*pi,1.0049393995505387*pi) q[9];
U1q(0.521917210807045*pi,1.149320044106756*pi) q[10];
U1q(0.695088276946483*pi,0.30158379958779236*pi) q[11];
U1q(0.743182413343542*pi,0.30195550078059963*pi) q[12];
U1q(0.560821099185411*pi,0.1062284867378187*pi) q[13];
U1q(0.480438742904046*pi,0.007282181014252487*pi) q[14];
U1q(0.412660032020303*pi,1.8217332603706327*pi) q[15];
U1q(0.323347115433544*pi,1.7892692978132416*pi) q[16];
U1q(0.753401000764193*pi,1.31368533196395*pi) q[17];
U1q(0.226830904771874*pi,0.8663381808672899*pi) q[18];
U1q(0.912757723408799*pi,1.98284817611344*pi) q[19];
U1q(0.872366222158116*pi,1.41528776838629*pi) q[20];
U1q(0.54098222578645*pi,0.09265007726197005*pi) q[21];
U1q(0.478589573640413*pi,1.1571524462481402*pi) q[22];
U1q(0.0697988164683654*pi,1.3977428251092148*pi) q[23];
U1q(0.567316350373878*pi,0.6453682439046702*pi) q[24];
U1q(0.550856808661834*pi,1.5943508207901047*pi) q[25];
U1q(0.482296790945962*pi,1.8767830648145045*pi) q[26];
U1q(0.503680936924454*pi,0.10468386905044014*pi) q[27];
U1q(0.232217027353613*pi,0.35079707935036897*pi) q[28];
U1q(0.233824020139859*pi,0.45923935876072*pi) q[29];
U1q(0.752691368456527*pi,1.2966751474245903*pi) q[30];
U1q(0.21196521722612*pi,1.9945013266526503*pi) q[31];
U1q(0.707714578492683*pi,1.7747781526083797*pi) q[32];
U1q(0.672782853909848*pi,1.7006532831788634*pi) q[33];
U1q(0.296491101720711*pi,1.8245020684079005*pi) q[34];
U1q(0.75385053310838*pi,1.61054598086834*pi) q[35];
U1q(0.431464411539034*pi,0.5069987931920901*pi) q[36];
U1q(0.0906499920474325*pi,1.52187807837148*pi) q[37];
U1q(0.331165480060465*pi,0.4141077976232097*pi) q[38];
U1q(0.339776727022309*pi,1.546600251024011*pi) q[39];
RZZ(0.5*pi) q[0],q[19];
RZZ(0.5*pi) q[8],q[1];
RZZ(0.5*pi) q[2],q[15];
RZZ(0.5*pi) q[13],q[3];
RZZ(0.5*pi) q[4],q[38];
RZZ(0.5*pi) q[21],q[5];
RZZ(0.5*pi) q[26],q[6];
RZZ(0.5*pi) q[37],q[7];
RZZ(0.5*pi) q[9],q[22];
RZZ(0.5*pi) q[10],q[16];
RZZ(0.5*pi) q[11],q[31];
RZZ(0.5*pi) q[12],q[20];
RZZ(0.5*pi) q[14],q[36];
RZZ(0.5*pi) q[17],q[28];
RZZ(0.5*pi) q[18],q[30];
RZZ(0.5*pi) q[33],q[23];
RZZ(0.5*pi) q[34],q[24];
RZZ(0.5*pi) q[39],q[25];
RZZ(0.5*pi) q[27],q[35];
RZZ(0.5*pi) q[32],q[29];
U1q(0.75511852757908*pi,1.7846507858966296*pi) q[0];
U1q(0.136943764103372*pi,1.5612983775176819*pi) q[1];
U1q(0.104665632766337*pi,1.2467145239445658*pi) q[2];
U1q(0.191366603537865*pi,1.4092688677000202*pi) q[3];
U1q(0.236721205667935*pi,0.25016092921013033*pi) q[4];
U1q(0.331386069347204*pi,1.7236399238289355*pi) q[5];
U1q(0.297839101822223*pi,1.7475923102770103*pi) q[6];
U1q(0.781794538312876*pi,0.36178825571072704*pi) q[7];
U1q(0.485316486017204*pi,1.6971595645870936*pi) q[8];
U1q(0.839682281020954*pi,0.23538954143847857*pi) q[9];
U1q(0.59293604440753*pi,0.9228650433048857*pi) q[10];
U1q(0.721894503121888*pi,1.0124402138718023*pi) q[11];
U1q(0.253659195893073*pi,0.9579655991346492*pi) q[12];
U1q(0.606275859476082*pi,1.2338598717521485*pi) q[13];
U1q(0.315122259226187*pi,0.13827978788794226*pi) q[14];
U1q(0.129820397665606*pi,1.898489171494992*pi) q[15];
U1q(0.203238787916841*pi,0.6155451287643103*pi) q[16];
U1q(0.605589924551427*pi,1.5549267462089098*pi) q[17];
U1q(0.74620453126489*pi,0.22631102554429994*pi) q[18];
U1q(0.804633205618498*pi,0.6082950814129298*pi) q[19];
U1q(0.197733439502663*pi,1.2150165141918903*pi) q[20];
U1q(0.632272471089636*pi,0.6743862347315499*pi) q[21];
U1q(0.560106110607004*pi,1.3164322220616098*pi) q[22];
U1q(0.181328267154815*pi,0.020080354652914245*pi) q[23];
U1q(0.549846347511987*pi,1.38678906070212*pi) q[24];
U1q(0.593379195046913*pi,0.44866445191324456*pi) q[25];
U1q(0.479794216049693*pi,1.5591147743368952*pi) q[26];
U1q(0.752023377076989*pi,0.9763136362238702*pi) q[27];
U1q(0.102143033916732*pi,0.00903945024187891*pi) q[28];
U1q(0.555123737997507*pi,0.7794386576865602*pi) q[29];
U1q(0.657873686703714*pi,0.45526212456660975*pi) q[30];
U1q(0.489397841996096*pi,1.9450388404860206*pi) q[31];
U1q(0.508835292954861*pi,1.4338142311557496*pi) q[32];
U1q(0.708893762086117*pi,1.4315045186896933*pi) q[33];
U1q(0.447024547780694*pi,1.7853693312840306*pi) q[34];
U1q(0.624363298403807*pi,1.0851021199362298*pi) q[35];
U1q(0.484521519262535*pi,0.6326501964079396*pi) q[36];
U1q(0.5103718289167*pi,1.50683362112565*pi) q[37];
U1q(0.74630733299816*pi,1.2955160395829903*pi) q[38];
U1q(0.781091469561898*pi,1.0993134371912916*pi) q[39];
RZZ(0.5*pi) q[37],q[0];
RZZ(0.5*pi) q[26],q[1];
RZZ(0.5*pi) q[2],q[23];
RZZ(0.5*pi) q[3],q[24];
RZZ(0.5*pi) q[4],q[22];
RZZ(0.5*pi) q[5],q[6];
RZZ(0.5*pi) q[7],q[27];
RZZ(0.5*pi) q[8],q[20];
RZZ(0.5*pi) q[13],q[9];
RZZ(0.5*pi) q[10],q[11];
RZZ(0.5*pi) q[31],q[12];
RZZ(0.5*pi) q[17],q[14];
RZZ(0.5*pi) q[29],q[15];
RZZ(0.5*pi) q[16],q[36];
RZZ(0.5*pi) q[39],q[18];
RZZ(0.5*pi) q[34],q[19];
RZZ(0.5*pi) q[33],q[21];
RZZ(0.5*pi) q[28],q[25];
RZZ(0.5*pi) q[30],q[38];
RZZ(0.5*pi) q[32],q[35];
U1q(0.737264738936208*pi,0.7159267770891002*pi) q[0];
U1q(0.804727433688112*pi,1.9567378422524317*pi) q[1];
U1q(0.788018214248351*pi,0.9034846080061758*pi) q[2];
U1q(0.336070905599755*pi,1.77748222996766*pi) q[3];
U1q(0.471615583727447*pi,1.4792490901781594*pi) q[4];
U1q(0.221140245439133*pi,1.982039042774426*pi) q[5];
U1q(0.49791459008006*pi,1.5836282512037005*pi) q[6];
U1q(0.257060619727802*pi,0.3324576972559168*pi) q[7];
U1q(0.508700110069562*pi,1.8791258567480629*pi) q[8];
U1q(0.791330054192125*pi,1.9255154343031666*pi) q[9];
U1q(0.672295592713516*pi,0.6439013707837562*pi) q[10];
U1q(0.41365526612621*pi,1.534417343027842*pi) q[11];
U1q(0.404721220280773*pi,1.3584742083796382*pi) q[12];
U1q(0.845783992155018*pi,0.2824908129502388*pi) q[13];
U1q(0.100563183570871*pi,0.9205515264854824*pi) q[14];
U1q(0.560668228243148*pi,0.8017024620037123*pi) q[15];
U1q(0.857737295916377*pi,0.8272506457924003*pi) q[16];
U1q(0.769213620778103*pi,0.0835957634169997*pi) q[17];
U1q(0.767669095487192*pi,1.9697459601060494*pi) q[18];
U1q(0.641521685577509*pi,1.6068687814818094*pi) q[19];
U1q(0.45751843921914*pi,1.31441662125403*pi) q[20];
U1q(0.471500797994343*pi,1.7443306349051202*pi) q[21];
U1q(0.510862819625974*pi,1.4787446326613596*pi) q[22];
U1q(0.118042823201502*pi,0.08002197907715392*pi) q[23];
U1q(0.887682097988856*pi,0.3980091856421506*pi) q[24];
U1q(0.585481109429612*pi,1.262495493612895*pi) q[25];
U1q(0.558239351744333*pi,0.7337463220482157*pi) q[26];
U1q(0.24436946880586*pi,1.7471085076228903*pi) q[27];
U1q(0.167754196919917*pi,0.17190409891850855*pi) q[28];
U1q(0.602092193408554*pi,0.8887406867256704*pi) q[29];
U1q(0.458025091424597*pi,0.5405869891358401*pi) q[30];
U1q(0.381485638185671*pi,1.1381417775515992*pi) q[31];
U1q(0.479093369864329*pi,1.5447240659877206*pi) q[32];
U1q(0.448534299366177*pi,1.420236001568723*pi) q[33];
U1q(0.398529387700455*pi,1.906986911070172*pi) q[34];
U1q(0.19255651763582*pi,1.3100811508515608*pi) q[35];
U1q(0.706888577737056*pi,0.42554202358332915*pi) q[36];
U1q(0.243792406850628*pi,1.9618995984109002*pi) q[37];
U1q(0.597233545664643*pi,0.061215573393399225*pi) q[38];
U1q(0.321442430567466*pi,0.5241009662054008*pi) q[39];
RZZ(0.5*pi) q[0],q[30];
RZZ(0.5*pi) q[1],q[18];
RZZ(0.5*pi) q[17],q[2];
RZZ(0.5*pi) q[3],q[36];
RZZ(0.5*pi) q[13],q[4];
RZZ(0.5*pi) q[5],q[38];
RZZ(0.5*pi) q[33],q[6];
RZZ(0.5*pi) q[7],q[29];
RZZ(0.5*pi) q[8],q[39];
RZZ(0.5*pi) q[9],q[12];
RZZ(0.5*pi) q[10],q[19];
RZZ(0.5*pi) q[11],q[24];
RZZ(0.5*pi) q[14],q[25];
RZZ(0.5*pi) q[15],q[23];
RZZ(0.5*pi) q[21],q[16];
RZZ(0.5*pi) q[35],q[20];
RZZ(0.5*pi) q[32],q[22];
RZZ(0.5*pi) q[37],q[26];
RZZ(0.5*pi) q[34],q[27];
RZZ(0.5*pi) q[28],q[31];
U1q(0.377321236117183*pi,1.0046966544344205*pi) q[0];
U1q(0.499924018831702*pi,0.2106129533290808*pi) q[1];
U1q(0.608603043437677*pi,1.7290901955676468*pi) q[2];
U1q(0.351409283625219*pi,1.053707373314671*pi) q[3];
U1q(0.123356662520678*pi,1.4934814598092991*pi) q[4];
U1q(0.740912227778095*pi,0.4861203179350948*pi) q[5];
U1q(0.578735440776485*pi,1.0838391650624004*pi) q[6];
U1q(0.320179203334557*pi,0.10597716622043674*pi) q[7];
U1q(0.670019080635817*pi,0.9722731627482037*pi) q[8];
U1q(0.0237420020803428*pi,0.9884282443833072*pi) q[9];
U1q(0.11187902592333*pi,0.8610888013364555*pi) q[10];
U1q(0.826479662243379*pi,1.5734725800927531*pi) q[11];
U1q(0.815049680545087*pi,0.12020522538891854*pi) q[12];
U1q(0.50536738517601*pi,0.7066354865764195*pi) q[13];
U1q(0.477587049613478*pi,0.763092000514682*pi) q[14];
U1q(0.337119225590857*pi,1.0307463613046135*pi) q[15];
U1q(0.619561529975366*pi,0.7907303425540615*pi) q[16];
U1q(0.688180346890825*pi,1.4424587416961607*pi) q[17];
U1q(0.360086968240064*pi,0.9204498027357992*pi) q[18];
U1q(0.349782337771779*pi,1.8175266065501*pi) q[19];
U1q(0.373830751039668*pi,1.4608804078343596*pi) q[20];
U1q(0.633716785420085*pi,0.25406651015465975*pi) q[21];
U1q(0.811099306438878*pi,1.9694848749554996*pi) q[22];
U1q(0.162103294397723*pi,1.0733205299448851*pi) q[23];
U1q(0.819451248870891*pi,0.7574902561283494*pi) q[24];
U1q(0.156229129603371*pi,0.33408874157352386*pi) q[25];
U1q(0.349445031614865*pi,1.0165760160793056*pi) q[26];
U1q(0.707059977615821*pi,0.23033821598343973*pi) q[27];
U1q(0.363493257131025*pi,1.1428309941003079*pi) q[28];
U1q(0.884009527523184*pi,0.8419314553653994*pi) q[29];
U1q(0.229521850671081*pi,1.4337061649088003*pi) q[30];
U1q(0.726435800597454*pi,1.5603370375522*pi) q[31];
U1q(0.420568299730941*pi,0.36420803275339964*pi) q[32];
U1q(0.528544588438113*pi,1.258206761520583*pi) q[33];
U1q(0.518331402812274*pi,0.644902223657672*pi) q[34];
U1q(0.263008209009778*pi,1.1598818663662005*pi) q[35];
U1q(0.312966177283836*pi,0.8215613342516992*pi) q[36];
U1q(0.669834394444615*pi,0.1263584199305008*pi) q[37];
U1q(0.463777822833037*pi,1.5258064087412997*pi) q[38];
U1q(0.74581637807079*pi,1.0724107152181208*pi) q[39];
RZZ(0.5*pi) q[0],q[18];
RZZ(0.5*pi) q[21],q[1];
RZZ(0.5*pi) q[2],q[6];
RZZ(0.5*pi) q[3],q[11];
RZZ(0.5*pi) q[4],q[9];
RZZ(0.5*pi) q[5],q[15];
RZZ(0.5*pi) q[7],q[17];
RZZ(0.5*pi) q[10],q[8];
RZZ(0.5*pi) q[39],q[12];
RZZ(0.5*pi) q[13],q[33];
RZZ(0.5*pi) q[14],q[22];
RZZ(0.5*pi) q[16],q[31];
RZZ(0.5*pi) q[25],q[19];
RZZ(0.5*pi) q[23],q[20];
RZZ(0.5*pi) q[24],q[38];
RZZ(0.5*pi) q[34],q[26];
RZZ(0.5*pi) q[29],q[27];
RZZ(0.5*pi) q[28],q[35];
RZZ(0.5*pi) q[30],q[36];
RZZ(0.5*pi) q[32],q[37];
U1q(0.808610266160144*pi,0.31535455681859936*pi) q[0];
U1q(0.925923949339343*pi,1.519409303367782*pi) q[1];
U1q(0.187767532845968*pi,1.027527383069847*pi) q[2];
U1q(0.859099331296326*pi,1.9863393714772712*pi) q[3];
U1q(0.947752539948052*pi,0.04910359804470055*pi) q[4];
U1q(0.632509445927784*pi,1.2748425123206957*pi) q[5];
U1q(0.162375750133957*pi,1.2110896821392991*pi) q[6];
U1q(0.343780181118517*pi,1.378330366889557*pi) q[7];
U1q(0.490674282954795*pi,0.5637315683360029*pi) q[8];
U1q(0.7519231448419*pi,1.2003907747392066*pi) q[9];
U1q(0.104903480903947*pi,1.5742498256791553*pi) q[10];
U1q(0.125775390515537*pi,0.23443073769391276*pi) q[11];
U1q(0.0144845203143855*pi,0.30682002786060814*pi) q[12];
U1q(0.628869078562491*pi,0.6996061627286796*pi) q[13];
U1q(0.449305403178197*pi,0.7765307349532815*pi) q[14];
U1q(0.767377441573855*pi,1.8334784726231135*pi) q[15];
U1q(0.446560285501137*pi,1.3449238429090613*pi) q[16];
U1q(0.365383013716872*pi,1.0220566891534002*pi) q[17];
U1q(0.686446803245075*pi,1.3720387252863997*pi) q[18];
U1q(0.324416091774678*pi,0.6258783641081997*pi) q[19];
U1q(0.668068287768875*pi,0.23384336068949985*pi) q[20];
U1q(0.271038876348273*pi,1.3811834961248994*pi) q[21];
U1q(0.0333510374513643*pi,1.3693598607453108*pi) q[22];
U1q(0.519066495815*pi,1.0802555098749842*pi) q[23];
U1q(0.64979112383306*pi,1.5505240533540494*pi) q[24];
U1q(0.453241251306009*pi,0.0229182383925437*pi) q[25];
U1q(0.301798771877208*pi,0.5041997507974045*pi) q[26];
U1q(0.236386275522244*pi,0.47756042877792915*pi) q[27];
U1q(0.658306220191557*pi,0.5131527583559077*pi) q[28];
U1q(0.378773282089968*pi,1.3613366991717992*pi) q[29];
U1q(0.284884922285341*pi,0.2907916557508994*pi) q[30];
U1q(0.524070491300074*pi,1.8801487210052006*pi) q[31];
U1q(0.727372259895774*pi,1.0187194455024997*pi) q[32];
U1q(0.78126697458149*pi,0.7205995307395838*pi) q[33];
U1q(0.495057591308393*pi,0.6356643307789707*pi) q[34];
U1q(0.48079140996509*pi,1.4527921693177*pi) q[35];
U1q(0.899836611070715*pi,0.9776659118933999*pi) q[36];
U1q(0.648321456137306*pi,1.4970121388988993*pi) q[37];
U1q(0.384042757340124*pi,1.4895098575405008*pi) q[38];
U1q(0.410575871303262*pi,0.752968910663121*pi) q[39];
rz(0.1708933989479995*pi) q[0];
rz(2.962869718349719*pi) q[1];
rz(0.601316336715854*pi) q[2];
rz(1.3315072142881306*pi) q[3];
rz(3.8740346288872*pi) q[4];
rz(2.031113318703504*pi) q[5];
rz(0.18843515965379964*pi) q[6];
rz(3.2510399321916417*pi) q[7];
rz(2.461301313068196*pi) q[8];
rz(0.5598561923660927*pi) q[9];
rz(2.2744246848664442*pi) q[10];
rz(3.7122154854743883*pi) q[11];
rz(3.362664509985592*pi) q[12];
rz(1.4757604998127807*pi) q[13];
rz(2.279287155471918*pi) q[14];
rz(3.928123843765688*pi) q[15];
rz(3.845934275646039*pi) q[16];
rz(3.6661800374928006*pi) q[17];
rz(1.5375508764841008*pi) q[18];
rz(1.7496718557625002*pi) q[19];
rz(3.6647550339142008*pi) q[20];
rz(0.8164600205324994*pi) q[21];
rz(0.41455784665127027*pi) q[22];
rz(0.7238986999926151*pi) q[23];
rz(0.27850685796214947*pi) q[24];
rz(0.840962553374057*pi) q[25];
rz(0.48128072144039535*pi) q[26];
rz(2.918837053909*pi) q[27];
rz(3.507183066315692*pi) q[28];
rz(2.1311556632957007*pi) q[29];
rz(0.6795481136585*pi) q[30];
rz(2.4643564894969003*pi) q[31];
rz(3.5491843313548994*pi) q[32];
rz(0.9303388151612175*pi) q[33];
rz(0.015936168622928903*pi) q[34];
rz(1.8784164094155003*pi) q[35];
rz(0.9929970112203002*pi) q[36];
rz(1.8768007838626009*pi) q[37];
rz(2.1531081730772996*pi) q[38];
rz(0.14791280904868032*pi) q[39];
U1q(0.808610266160144*pi,1.486247955766663*pi) q[0];
U1q(0.925923949339343*pi,1.482279021717471*pi) q[1];
U1q(0.187767532845968*pi,0.628843719785718*pi) q[2];
U1q(0.859099331296326*pi,0.317846585765382*pi) q[3];
U1q(1.94775253994805*pi,0.923138226931943*pi) q[4];
U1q(1.63250944592778*pi,0.305955831024235*pi) q[5];
U1q(0.162375750133957*pi,0.399524841793059*pi) q[6];
U1q(1.34378018111852*pi,1.629370299081189*pi) q[7];
U1q(0.490674282954795*pi,0.0250328814042431*pi) q[8];
U1q(0.7519231448419*pi,0.760246967105299*pi) q[9];
U1q(0.104903480903947*pi,0.8486745105456599*pi) q[10];
U1q(1.12577539051554*pi,0.94664622316823*pi) q[11];
U1q(1.01448452031439*pi,0.66948453784619*pi) q[12];
U1q(0.628869078562491*pi,1.17536666254146*pi) q[13];
U1q(1.4493054031782*pi,0.0558178904251752*pi) q[14];
U1q(3.767377441573856*pi,0.761602316388794*pi) q[15];
U1q(1.44656028550114*pi,0.190858118555139*pi) q[16];
U1q(1.36538301371687*pi,1.68823672664617*pi) q[17];
U1q(0.686446803245075*pi,1.9095896017704945*pi) q[18];
U1q(1.32441609177468*pi,1.375550219870666*pi) q[19];
U1q(0.668068287768875*pi,0.8985983946037*pi) q[20];
U1q(0.271038876348273*pi,1.197643516657417*pi) q[21];
U1q(3.033351037451364*pi,0.78391770739658*pi) q[22];
U1q(1.519066495815*pi,0.804154209867606*pi) q[23];
U1q(0.64979112383306*pi,0.829030911316164*pi) q[24];
U1q(1.45324125130601*pi,1.863880791766666*pi) q[25];
U1q(0.301798771877208*pi,1.985480472237773*pi) q[26];
U1q(3.236386275522244*pi,0.396397482686912*pi) q[27];
U1q(0.658306220191557*pi,1.02033582467165*pi) q[28];
U1q(1.37877328208997*pi,0.492492362467506*pi) q[29];
U1q(0.284884922285341*pi,1.9703397694093863*pi) q[30];
U1q(1.52407049130007*pi,1.344505210502168*pi) q[31];
U1q(1.72737225989577*pi,1.567903776857454*pi) q[32];
U1q(3.781266974581491*pi,0.650938345900844*pi) q[33];
U1q(1.49505759130839*pi,1.651600499401885*pi) q[34];
U1q(0.48079140996509*pi,0.331208578733262*pi) q[35];
U1q(1.89983661107072*pi,0.970662923113736*pi) q[36];
U1q(0.648321456137306*pi,0.373812922761465*pi) q[37];
U1q(0.384042757340124*pi,0.64261803061782*pi) q[38];
U1q(1.41057587130326*pi,1.9008817197117631*pi) q[39];
RZZ(0.5*pi) q[0],q[18];
RZZ(0.5*pi) q[21],q[1];
RZZ(0.5*pi) q[2],q[6];
RZZ(0.5*pi) q[3],q[11];
RZZ(0.5*pi) q[4],q[9];
RZZ(0.5*pi) q[5],q[15];
RZZ(0.5*pi) q[7],q[17];
RZZ(0.5*pi) q[10],q[8];
RZZ(0.5*pi) q[39],q[12];
RZZ(0.5*pi) q[13],q[33];
RZZ(0.5*pi) q[14],q[22];
RZZ(0.5*pi) q[16],q[31];
RZZ(0.5*pi) q[25],q[19];
RZZ(0.5*pi) q[23],q[20];
RZZ(0.5*pi) q[24],q[38];
RZZ(0.5*pi) q[34],q[26];
RZZ(0.5*pi) q[29],q[27];
RZZ(0.5*pi) q[28],q[35];
RZZ(0.5*pi) q[30],q[36];
RZZ(0.5*pi) q[32],q[37];
U1q(0.377321236117183*pi,1.1755900533824701*pi) q[0];
U1q(0.499924018831702*pi,1.173482671678776*pi) q[1];
U1q(1.60860304343768*pi,0.3304065322835401*pi) q[2];
U1q(0.351409283625219*pi,1.385214587602787*pi) q[3];
U1q(1.12335666252068*pi,1.4787603651673447*pi) q[4];
U1q(3.259087772221905*pi,1.0946780254098356*pi) q[5];
U1q(1.57873544077649*pi,1.27227432471615*pi) q[6];
U1q(3.679820796665443*pi,0.9017234997502712*pi) q[7];
U1q(1.67001908063582*pi,1.433574475816386*pi) q[8];
U1q(0.0237420020803428*pi,0.54828443674933*pi) q[9];
U1q(1.11187902592333*pi,1.13551348620298*pi) q[10];
U1q(1.82647966224338*pi,1.6076043807693714*pi) q[11];
U1q(1.81504968054509*pi,1.8560993403179047*pi) q[12];
U1q(0.50536738517601*pi,1.182395986389204*pi) q[13];
U1q(3.522412950386522*pi,0.06925662486376538*pi) q[14];
U1q(1.33711922559086*pi,0.5643344277072897*pi) q[15];
U1q(3.3804384700246333*pi,0.7450516189101566*pi) q[16];
U1q(3.688180346890825*pi,0.2678346741034001*pi) q[17];
U1q(0.360086968240064*pi,0.45800067921998*pi) q[18];
U1q(3.650217662228221*pi,0.18390197742882042*pi) q[19];
U1q(1.37383075103967*pi,0.12563544174860009*pi) q[20];
U1q(0.633716785420085*pi,1.07052653068714*pi) q[21];
U1q(3.188900693561123*pi,0.18379269318638958*pi) q[22];
U1q(3.837896705602277*pi,1.811089189797634*pi) q[23];
U1q(0.819451248870891*pi,1.03599711409052*pi) q[24];
U1q(3.843770870396629*pi,1.552710288585707*pi) q[25];
U1q(1.34944503161487*pi,0.49785673751974*pi) q[26];
U1q(3.292940022384179*pi,0.6436196954814125*pi) q[27];
U1q(0.363493257131025*pi,0.650014060416049*pi) q[28];
U1q(1.88400952752318*pi,0.011897606273962857*pi) q[29];
U1q(1.22952185067108*pi,1.113254278567318*pi) q[30];
U1q(1.72643580059745*pi,0.6643168939552244*pi) q[31];
U1q(3.579431700269059*pi,1.22241518960655*pi) q[32];
U1q(3.471455411561886*pi,1.1133311151198577*pi) q[33];
U1q(1.51833140281227*pi,1.6423626065231869*pi) q[34];
U1q(0.263008209009778*pi,1.03829827578174*pi) q[35];
U1q(1.31296617728384*pi,1.1267675007554943*pi) q[36];
U1q(0.669834394444615*pi,1.00315920379306*pi) q[37];
U1q(1.46377782283304*pi,1.6789145818185598*pi) q[38];
U1q(1.74581637807079*pi,0.5814399151567231*pi) q[39];
RZZ(0.5*pi) q[0],q[30];
RZZ(0.5*pi) q[1],q[18];
RZZ(0.5*pi) q[17],q[2];
RZZ(0.5*pi) q[3],q[36];
RZZ(0.5*pi) q[13],q[4];
RZZ(0.5*pi) q[5],q[38];
RZZ(0.5*pi) q[33],q[6];
RZZ(0.5*pi) q[7],q[29];
RZZ(0.5*pi) q[8],q[39];
RZZ(0.5*pi) q[9],q[12];
RZZ(0.5*pi) q[10],q[19];
RZZ(0.5*pi) q[11],q[24];
RZZ(0.5*pi) q[14],q[25];
RZZ(0.5*pi) q[15],q[23];
RZZ(0.5*pi) q[21],q[16];
RZZ(0.5*pi) q[35],q[20];
RZZ(0.5*pi) q[32],q[22];
RZZ(0.5*pi) q[37],q[26];
RZZ(0.5*pi) q[34],q[27];
RZZ(0.5*pi) q[28],q[31];
U1q(0.737264738936208*pi,0.8868201760371499*pi) q[0];
U1q(0.804727433688112*pi,1.919607560602127*pi) q[1];
U1q(3.2119817857516493*pi,0.15601211984501717*pi) q[2];
U1q(1.33607090559975*pi,1.1089894442557702*pi) q[3];
U1q(1.47161558372745*pi,0.4645279955361618*pi) q[4];
U1q(3.778859754560867*pi,0.5987593005705377*pi) q[5];
U1q(3.502085409919939*pi,0.7724852385748825*pi) q[6];
U1q(3.742939380272197*pi,0.6752429687147914*pi) q[7];
U1q(3.491299889930438*pi,1.5267217818165044*pi) q[8];
U1q(0.791330054192125*pi,0.48537162666923006*pi) q[9];
U1q(3.327704407286484*pi,1.3527009167557331*pi) q[10];
U1q(1.41365526612621*pi,0.5685491437044612*pi) q[11];
U1q(0.404721220280773*pi,0.09436832330862477*pi) q[12];
U1q(0.845783992155018*pi,1.75825131276302*pi) q[13];
U1q(3.899436816429129*pi,0.9117970988929565*pi) q[14];
U1q(3.560668228243148*pi,0.3352905284063956*pi) q[15];
U1q(1.85773729591638*pi,1.7085313156718342*pi) q[16];
U1q(0.769213620778103*pi,0.9089716958242371*pi) q[17];
U1q(0.767669095487192*pi,0.50729683659018*pi) q[18];
U1q(3.358478314422491*pi,1.3945598024970605*pi) q[19];
U1q(3.54248156078086*pi,1.2720992283289403*pi) q[20];
U1q(0.471500797994343*pi,0.5607906554376001*pi) q[21];
U1q(3.489137180374026*pi,1.6745329354805336*pi) q[22];
U1q(1.1180428232015*pi,1.804387740665411*pi) q[23];
U1q(1.88768209798886*pi,1.6765160436043*pi) q[24];
U1q(3.4145188905703883*pi,1.624303536546339*pi) q[25];
U1q(1.55823935174433*pi,0.7806864315508806*pi) q[26];
U1q(3.75563053119414*pi,1.1268494038419625*pi) q[27];
U1q(1.16775419691992*pi,0.6790871652342401*pi) q[28];
U1q(0.602092193408554*pi,0.05870683763426182*pi) q[29];
U1q(3.541974908575403*pi,1.0063734543402605*pi) q[30];
U1q(0.381485638185671*pi,1.2421216339546044*pi) q[31];
U1q(3.520906630135671*pi,1.04189915637224*pi) q[32];
U1q(3.551465700633823*pi,1.9513018750717395*pi) q[33];
U1q(0.398529387700455*pi,0.904447293935722*pi) q[34];
U1q(0.19255651763582*pi,1.1884975602670802*pi) q[35];
U1q(1.70688857773706*pi,1.7307481900871682*pi) q[36];
U1q(1.24379240685063*pi,0.83870038227346*pi) q[37];
U1q(3.402766454335357*pi,1.1435054171664483*pi) q[38];
U1q(1.32144243056747*pi,0.03313016614398001*pi) q[39];
RZZ(0.5*pi) q[37],q[0];
RZZ(0.5*pi) q[26],q[1];
RZZ(0.5*pi) q[2],q[23];
RZZ(0.5*pi) q[3],q[24];
RZZ(0.5*pi) q[4],q[22];
RZZ(0.5*pi) q[5],q[6];
RZZ(0.5*pi) q[7],q[27];
RZZ(0.5*pi) q[8],q[20];
RZZ(0.5*pi) q[13],q[9];
RZZ(0.5*pi) q[10],q[11];
RZZ(0.5*pi) q[31],q[12];
RZZ(0.5*pi) q[17],q[14];
RZZ(0.5*pi) q[29],q[15];
RZZ(0.5*pi) q[16],q[36];
RZZ(0.5*pi) q[39],q[18];
RZZ(0.5*pi) q[34],q[19];
RZZ(0.5*pi) q[33],q[21];
RZZ(0.5*pi) q[28],q[25];
RZZ(0.5*pi) q[30],q[38];
RZZ(0.5*pi) q[32],q[35];
U1q(0.75511852757908*pi,1.9555441848446802*pi) q[0];
U1q(0.136943764103372*pi,0.52416809586737*pi) q[1];
U1q(3.895334367233661*pi,1.8127822039066221*pi) q[2];
U1q(3.191366603537865*pi,0.4772028065234135*pi) q[3];
U1q(1.23672120566793*pi,0.6936161565041923*pi) q[4];
U1q(1.3313860693472*pi,0.8571584195160281*pi) q[5];
U1q(1.29783910182222*pi,0.608521179501539*pi) q[6];
U1q(1.78179453831288*pi,0.6459124102599816*pi) q[7];
U1q(3.514683513982796*pi,0.7086880739774726*pi) q[8];
U1q(1.83968228102095*pi,1.7952457338045398*pi) q[9];
U1q(3.592936044407529*pi,1.0737372442345956*pi) q[10];
U1q(3.278105496878111*pi,1.0905262728605014*pi) q[11];
U1q(1.25365919589307*pi,0.6938597140636347*pi) q[12];
U1q(1.60627585947608*pi,1.70962037156493*pi) q[13];
U1q(3.315122259226187*pi,0.6940688374904952*pi) q[14];
U1q(1.12982039766561*pi,1.238503818915143*pi) q[15];
U1q(0.203238787916841*pi,1.4968257986437443*pi) q[16];
U1q(0.605589924551427*pi,0.3803026786161501*pi) q[17];
U1q(0.74620453126489*pi,0.7638619020284301*pi) q[18];
U1q(1.8046332056185*pi,1.3931335025659415*pi) q[19];
U1q(3.802266560497337*pi,0.37149933539107005*pi) q[20];
U1q(0.632272471089636*pi,1.4908462552640298*pi) q[21];
U1q(3.4398938893929962*pi,0.8368453460802736*pi) q[22];
U1q(0.181328267154815*pi,1.74444611624117*pi) q[23];
U1q(3.450153652488013*pi,0.6877361685443306*pi) q[24];
U1q(3.406620804953087*pi,1.438134578245983*pi) q[25];
U1q(0.479794216049693*pi,0.6060548838395605*pi) q[26];
U1q(3.247976622923011*pi,0.8976442752409725*pi) q[27];
U1q(3.897856966083267*pi,0.8419518139108737*pi) q[28];
U1q(1.55512373799751*pi,1.9494048085951599*pi) q[29];
U1q(3.342126313296286*pi,0.09169831890950148*pi) q[30];
U1q(3.4893978419960963*pi,1.0490186968890542*pi) q[31];
U1q(3.4911647070451393*pi,1.1528089912042203*pi) q[32];
U1q(1.70889376208612*pi,0.9400333579507698*pi) q[33];
U1q(1.44702454778069*pi,0.7828297141495719*pi) q[34];
U1q(1.62436329840381*pi,0.9635185293517501*pi) q[35];
U1q(3.515478480737465*pi,0.523640017262561*pi) q[36];
U1q(3.4896281710832993*pi,0.2937663595587048*pi) q[37];
U1q(3.25369266700184*pi,0.9092049509768385*pi) q[38];
U1q(1.7810914695619*pi,1.4579176951580948*pi) q[39];
RZZ(0.5*pi) q[0],q[19];
RZZ(0.5*pi) q[8],q[1];
RZZ(0.5*pi) q[2],q[15];
RZZ(0.5*pi) q[13],q[3];
RZZ(0.5*pi) q[4],q[38];
RZZ(0.5*pi) q[21],q[5];
RZZ(0.5*pi) q[26],q[6];
RZZ(0.5*pi) q[37],q[7];
RZZ(0.5*pi) q[9],q[22];
RZZ(0.5*pi) q[10],q[16];
RZZ(0.5*pi) q[11],q[31];
RZZ(0.5*pi) q[12],q[20];
RZZ(0.5*pi) q[14],q[36];
RZZ(0.5*pi) q[17],q[28];
RZZ(0.5*pi) q[18],q[30];
RZZ(0.5*pi) q[33],q[23];
RZZ(0.5*pi) q[34],q[24];
RZZ(0.5*pi) q[39],q[25];
RZZ(0.5*pi) q[27],q[35];
RZZ(0.5*pi) q[32],q[29];
U1q(1.41816673819257*pi,0.9269601841050501*pi) q[0];
U1q(0.379912740237965*pi,1.34759771409932*pi) q[1];
U1q(3.7409001472112378*pi,1.2820977217434613*pi) q[2];
U1q(0.451170510175043*pi,0.7709938538121133*pi) q[3];
U1q(0.401682885840488*pi,0.6496302820520721*pi) q[4];
U1q(0.177218997336359*pi,1.6849822003032262*pi) q[5];
U1q(0.737402852534316*pi,0.05179881457933877*pi) q[6];
U1q(0.85650367923329*pi,1.8286587490340418*pi) q[7];
U1q(1.22545065881387*pi,1.6505980883474707*pi) q[8];
U1q(3.153120520834518*pi,0.02569587569247833*pi) q[9];
U1q(0.521917210807045*pi,0.3001922450364658*pi) q[10];
U1q(3.304911723053517*pi,0.8013826871445113*pi) q[11];
U1q(3.256817586656458*pi,1.3498698124176745*pi) q[12];
U1q(3.439178900814588*pi,1.83725175657927*pi) q[13];
U1q(0.480438742904046*pi,0.5630712306168052*pi) q[14];
U1q(0.412660032020303*pi,1.161747907790783*pi) q[15];
U1q(0.323347115433544*pi,1.6705499676926747*pi) q[16];
U1q(1.75340100076419*pi,0.13906126437118016*pi) q[17];
U1q(0.226830904771874*pi,1.4038890573514298*pi) q[18];
U1q(3.912757723408799*pi,1.7676865972664415*pi) q[19];
U1q(3.872366222158116*pi,0.17122808119667776*pi) q[20];
U1q(0.54098222578645*pi,1.9091100977944002*pi) q[21];
U1q(3.521410426359587*pi,1.9961251218937441*pi) q[22];
U1q(1.06979881646837*pi,0.12210858669746694*pi) q[23];
U1q(1.56731635037388*pi,1.429156985341792*pi) q[24];
U1q(3.449143191338166*pi,0.2924482093691332*pi) q[25];
U1q(1.48229679094596*pi,1.9237231743171703*pi) q[26];
U1q(3.496319063075545*pi,0.7692740424144023*pi) q[27];
U1q(1.23221702735361*pi,1.500194184802389*pi) q[28];
U1q(3.76617597986014*pi,0.26960410752099806*pi) q[29];
U1q(3.247308631543473*pi,1.2502852960515143*pi) q[30];
U1q(1.21196521722612*pi,1.9995562107224218*pi) q[31];
U1q(3.2922854215073167*pi,1.8118450697515902*pi) q[32];
U1q(1.67278285390985*pi,0.20918212243993783*pi) q[33];
U1q(3.703508898279289*pi,0.7436969770257122*pi) q[34];
U1q(3.246149466891619*pi,0.43807466841963993*pi) q[35];
U1q(3.431464411539035*pi,1.6492914204784095*pi) q[36];
U1q(1.09064999204743*pi,1.2787219023128813*pi) q[37];
U1q(1.33116548006046*pi,0.79061319293662*pi) q[38];
U1q(0.339776727022309*pi,1.9052045089908143*pi) q[39];
RZZ(0.5*pi) q[0],q[6];
RZZ(0.5*pi) q[13],q[1];
RZZ(0.5*pi) q[2],q[11];
RZZ(0.5*pi) q[3],q[35];
RZZ(0.5*pi) q[4],q[12];
RZZ(0.5*pi) q[5],q[28];
RZZ(0.5*pi) q[32],q[7];
RZZ(0.5*pi) q[8],q[31];
RZZ(0.5*pi) q[9],q[38];
RZZ(0.5*pi) q[10],q[18];
RZZ(0.5*pi) q[33],q[14];
RZZ(0.5*pi) q[15],q[19];
RZZ(0.5*pi) q[23],q[16];
RZZ(0.5*pi) q[17],q[22];
RZZ(0.5*pi) q[27],q[20];
RZZ(0.5*pi) q[21],q[30];
RZZ(0.5*pi) q[24],q[26];
RZZ(0.5*pi) q[34],q[25];
RZZ(0.5*pi) q[37],q[29];
RZZ(0.5*pi) q[39],q[36];
U1q(3.454181777530582*pi,1.467348799305658*pi) q[0];
U1q(0.485903913483886*pi,0.7570226488006999*pi) q[1];
U1q(3.769819913733584*pi,1.172771214432771*pi) q[2];
U1q(3.757273387052579*pi,0.6322467243962437*pi) q[3];
U1q(0.272541853333136*pi,1.3959544804155222*pi) q[4];
U1q(1.26368544161967*pi,0.6787435752092363*pi) q[5];
U1q(0.586041343145436*pi,0.47601355333514284*pi) q[6];
U1q(3.763189444957732*pi,0.6179974030909214*pi) q[7];
U1q(0.66093585699897*pi,1.5513788829866808*pi) q[8];
U1q(0.512327423470307*pi,0.06608008268082788*pi) q[9];
U1q(1.45907774953721*pi,1.788427061760057*pi) q[10];
U1q(3.099875014591999*pi,1.988335419236261*pi) q[11];
U1q(3.643376965246777*pi,0.22035059937238444*pi) q[12];
U1q(3.372631507843678*pi,0.41627588735723986*pi) q[13];
U1q(1.46448409326872*pi,0.1489269202755471*pi) q[14];
U1q(0.269809171166897*pi,1.6215671304038235*pi) q[15];
U1q(0.21270511153917*pi,1.909394042860594*pi) q[16];
U1q(3.696452212399626*pi,1.039659717766274*pi) q[17];
U1q(0.319682833833*pi,1.6875788569627401*pi) q[18];
U1q(3.29636030008226*pi,1.9495226097262988*pi) q[19];
U1q(1.87324254398966*pi,1.0607239582882475*pi) q[20];
U1q(1.44739175400429*pi,0.49751390811080043*pi) q[21];
U1q(3.127556255689*pi,0.6076983767917712*pi) q[22];
U1q(3.601794009442203*pi,0.3536200813850203*pi) q[23];
U1q(1.21664512015945*pi,0.6404883019421925*pi) q[24];
U1q(3.3885097619096243*pi,0.4870531119413326*pi) q[25];
U1q(3.571711629621662*pi,0.19889254513382149*pi) q[26];
U1q(1.53106071543453*pi,0.74917911910354*pi) q[27];
U1q(0.825239720217543*pi,0.008904611953498787*pi) q[28];
U1q(1.30269822035848*pi,0.012112932348401717*pi) q[29];
U1q(1.77234571734207*pi,0.15404994039098108*pi) q[30];
U1q(1.19553403953682*pi,0.7819606325148314*pi) q[31];
U1q(1.55363204456512*pi,0.9665273036970952*pi) q[32];
U1q(3.696672570929659*pi,1.1142435310407015*pi) q[33];
U1q(3.462306148984088*pi,1.3435269353281245*pi) q[34];
U1q(1.16459300656166*pi,0.89561545632125*pi) q[35];
U1q(1.52010878966808*pi,1.8507680718008914*pi) q[36];
U1q(0.927853474868702*pi,1.5021975795078015*pi) q[37];
U1q(0.260191697535144*pi,0.6115022799016092*pi) q[38];
U1q(0.387992120290644*pi,0.7294113052638647*pi) q[39];
RZZ(0.5*pi) q[2],q[0];
RZZ(0.5*pi) q[1],q[35];
RZZ(0.5*pi) q[3],q[19];
RZZ(0.5*pi) q[4],q[27];
RZZ(0.5*pi) q[29],q[5];
RZZ(0.5*pi) q[30],q[6];
RZZ(0.5*pi) q[7],q[12];
RZZ(0.5*pi) q[33],q[8];
RZZ(0.5*pi) q[9],q[28];
RZZ(0.5*pi) q[10],q[34];
RZZ(0.5*pi) q[11],q[20];
RZZ(0.5*pi) q[13],q[16];
RZZ(0.5*pi) q[14],q[26];
RZZ(0.5*pi) q[15],q[18];
RZZ(0.5*pi) q[17],q[36];
RZZ(0.5*pi) q[37],q[21];
RZZ(0.5*pi) q[22],q[39];
RZZ(0.5*pi) q[23],q[31];
RZZ(0.5*pi) q[32],q[24];
RZZ(0.5*pi) q[25],q[38];
U1q(3.0635961186697838*pi,1.0591470907672083*pi) q[0];
U1q(0.673576839471694*pi,0.1862480011478702*pi) q[1];
U1q(1.80276597386224*pi,1.2138298013935445*pi) q[2];
U1q(3.514489753066892*pi,0.43009383001272994*pi) q[3];
U1q(0.124918459652875*pi,1.2705636850037423*pi) q[4];
U1q(1.04947923171976*pi,0.24315143485788804*pi) q[5];
U1q(0.487808319006538*pi,1.501388613821609*pi) q[6];
U1q(1.53631820852483*pi,0.44073515359613236*pi) q[7];
U1q(0.387754284131565*pi,1.3974387339858767*pi) q[8];
U1q(0.390979829076419*pi,0.281433791678948*pi) q[9];
U1q(1.72653830137214*pi,0.6923581647809556*pi) q[10];
U1q(1.52496632002832*pi,1.3641171798507825*pi) q[11];
U1q(1.75678786056527*pi,0.39477630290917265*pi) q[12];
U1q(1.5523950488547*pi,1.8270030086845161*pi) q[13];
U1q(1.28048877310633*pi,1.8598689720323593*pi) q[14];
U1q(0.0912925810672765*pi,1.684022631518964*pi) q[15];
U1q(0.77610087016224*pi,0.4541423280561938*pi) q[16];
U1q(1.45375796645325*pi,1.6234326073223886*pi) q[17];
U1q(0.552985125170573*pi,0.026570829570180088*pi) q[18];
U1q(0.492444730307162*pi,0.43172628539986846*pi) q[19];
U1q(1.49556602592702*pi,0.17279952946651655*pi) q[20];
U1q(3.467167837466286*pi,1.5487833994696398*pi) q[21];
U1q(0.792567110097512*pi,1.2393242576175512*pi) q[22];
U1q(1.36532839301504*pi,1.8275710280067985*pi) q[23];
U1q(1.35514292641604*pi,1.5510465821176136*pi) q[24];
U1q(1.60904669481523*pi,1.6239066261809878*pi) q[25];
U1q(1.15998815337567*pi,1.7180842708533675*pi) q[26];
U1q(0.819775935735513*pi,1.161087390121629*pi) q[27];
U1q(0.258585458562174*pi,0.7757859206689188*pi) q[28];
U1q(0.713784365367372*pi,0.3098650041422517*pi) q[29];
U1q(0.408786441112893*pi,1.1544011065005733*pi) q[30];
U1q(1.65894948392087*pi,1.6011064409698879*pi) q[31];
U1q(0.708731682734186*pi,1.4170509045907647*pi) q[32];
U1q(1.62171208041534*pi,1.0447952280690147*pi) q[33];
U1q(1.594360204891*pi,0.8284899140116335*pi) q[34];
U1q(0.694856781051437*pi,0.5154476362787097*pi) q[35];
U1q(1.61475059010873*pi,1.0158770112787572*pi) q[36];
U1q(0.626069605237655*pi,1.3568550831819008*pi) q[37];
U1q(0.279100531754125*pi,0.16173021531676923*pi) q[38];
U1q(0.448242003847875*pi,1.8093563693117236*pi) q[39];
rz(0.9408529092327917*pi) q[0];
rz(3.81375199885213*pi) q[1];
rz(0.7861701986064555*pi) q[2];
rz(3.56990616998727*pi) q[3];
rz(2.7294363149962577*pi) q[4];
rz(1.756848565142112*pi) q[5];
rz(0.49861138617839096*pi) q[6];
rz(3.5592648464038676*pi) q[7];
rz(0.6025612660141233*pi) q[8];
rz(3.718566208321052*pi) q[9];
rz(1.3076418352190444*pi) q[10];
rz(0.6358828201492175*pi) q[11];
rz(3.6052236970908274*pi) q[12];
rz(2.172996991315484*pi) q[13];
rz(0.1401310279676406*pi) q[14];
rz(0.3159773684810361*pi) q[15];
rz(3.545857671943806*pi) q[16];
rz(2.3765673926776114*pi) q[17];
rz(1.97342917042982*pi) q[18];
rz(3.5682737146001315*pi) q[19];
rz(1.8272004705334834*pi) q[20];
rz(2.45121660053036*pi) q[21];
rz(0.7606757423824488*pi) q[22];
rz(0.1724289719932015*pi) q[23];
rz(2.4489534178823864*pi) q[24];
rz(0.3760933738190122*pi) q[25];
rz(2.2819157291466325*pi) q[26];
rz(2.838912609878371*pi) q[27];
rz(3.224214079331081*pi) q[28];
rz(1.6901349958577483*pi) q[29];
rz(0.8455988934994269*pi) q[30];
rz(0.39889355903011214*pi) q[31];
rz(0.5829490954092353*pi) q[32];
rz(2.955204771930985*pi) q[33];
rz(1.1715100859883665*pi) q[34];
rz(3.4845523637212903*pi) q[35];
rz(2.984122988721243*pi) q[36];
rz(2.643144916818099*pi) q[37];
rz(3.8382697846832308*pi) q[38];
rz(0.1906436306882764*pi) q[39];
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
measure q[16] -> c[16];
measure q[17] -> c[17];
measure q[18] -> c[18];
measure q[19] -> c[19];
measure q[20] -> c[20];
measure q[21] -> c[21];
measure q[22] -> c[22];
measure q[23] -> c[23];
measure q[24] -> c[24];
measure q[25] -> c[25];
measure q[26] -> c[26];
measure q[27] -> c[27];
measure q[28] -> c[28];
measure q[29] -> c[29];
measure q[30] -> c[30];
measure q[31] -> c[31];
measure q[32] -> c[32];
measure q[33] -> c[33];
measure q[34] -> c[34];
measure q[35] -> c[35];
measure q[36] -> c[36];
measure q[37] -> c[37];
measure q[38] -> c[38];
measure q[39] -> c[39];