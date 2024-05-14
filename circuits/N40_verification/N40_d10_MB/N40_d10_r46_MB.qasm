OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.322738425136568*pi,1.844411339962504*pi) q[0];
U1q(1.22740556140202*pi,0.9664080451518102*pi) q[1];
U1q(0.763853581822924*pi,0.439158752545703*pi) q[2];
U1q(0.42411318810458*pi,1.534706696708518*pi) q[3];
U1q(0.328155142044111*pi,0.8022233285560001*pi) q[4];
U1q(0.723616341809535*pi,0.095652109425244*pi) q[5];
U1q(0.458872013157779*pi,0.216142220025977*pi) q[6];
U1q(3.392592596792175*pi,1.0569038288997876*pi) q[7];
U1q(1.78873016279253*pi,0.623679450835884*pi) q[8];
U1q(1.87712504787906*pi,0.2586002911965816*pi) q[9];
U1q(1.47519698052643*pi,1.1898422212621704*pi) q[10];
U1q(1.06108626395657*pi,0.9068214922048243*pi) q[11];
U1q(0.284982340578374*pi,0.378647482640883*pi) q[12];
U1q(3.431619488303891*pi,0.992308106345353*pi) q[13];
U1q(1.17452970265903*pi,1.266852424266217*pi) q[14];
U1q(0.482898704837826*pi,0.734650457926743*pi) q[15];
U1q(3.394141747398708*pi,1.286684671640451*pi) q[16];
U1q(0.197257499186395*pi,0.173912012297605*pi) q[17];
U1q(1.72489867254521*pi,0.09836772332553911*pi) q[18];
U1q(0.233882817818338*pi,1.806026280682966*pi) q[19];
U1q(1.4670463418874*pi,0.6970382480687294*pi) q[20];
U1q(1.35728790354968*pi,1.414629102238118*pi) q[21];
U1q(1.65373596096344*pi,1.2735487351894585*pi) q[22];
U1q(0.922394268454152*pi,1.25106388821174*pi) q[23];
U1q(0.224001702570141*pi,1.03299677855348*pi) q[24];
U1q(0.554109768185238*pi,1.311952594182417*pi) q[25];
U1q(1.91588543251827*pi,1.724751945233832*pi) q[26];
U1q(0.422987213525969*pi,0.210457390142509*pi) q[27];
U1q(0.366933603614469*pi,1.2696079467537031*pi) q[28];
U1q(0.733396884291223*pi,1.322012681532247*pi) q[29];
U1q(1.53569273530802*pi,0.3524949619496901*pi) q[30];
U1q(1.73119010239702*pi,1.466662482727699*pi) q[31];
U1q(0.172945511628788*pi,0.772615332033881*pi) q[32];
U1q(0.586273835806525*pi,1.891013783434389*pi) q[33];
U1q(1.2164209514326*pi,0.8259443723128237*pi) q[34];
U1q(0.719122860429421*pi,1.34129338628912*pi) q[35];
U1q(0.210300563758592*pi,0.0778869628792963*pi) q[36];
U1q(0.889295208161124*pi,1.794343137425693*pi) q[37];
U1q(0.825921595027833*pi,0.778899806836438*pi) q[38];
U1q(1.80728853412691*pi,0.3300256618939907*pi) q[39];
RZZ(0.5*pi) q[0],q[3];
RZZ(0.5*pi) q[1],q[21];
RZZ(0.5*pi) q[2],q[20];
RZZ(0.5*pi) q[22],q[4];
RZZ(0.5*pi) q[5],q[24];
RZZ(0.5*pi) q[36],q[6];
RZZ(0.5*pi) q[7],q[15];
RZZ(0.5*pi) q[19],q[8];
RZZ(0.5*pi) q[29],q[9];
RZZ(0.5*pi) q[10],q[37];
RZZ(0.5*pi) q[11],q[28];
RZZ(0.5*pi) q[23],q[12];
RZZ(0.5*pi) q[13],q[31];
RZZ(0.5*pi) q[38],q[14];
RZZ(0.5*pi) q[16],q[25];
RZZ(0.5*pi) q[17],q[34];
RZZ(0.5*pi) q[26],q[18];
RZZ(0.5*pi) q[33],q[27];
RZZ(0.5*pi) q[30],q[35];
RZZ(0.5*pi) q[32],q[39];
U1q(0.502569050723224*pi,0.9323724727310001*pi) q[0];
U1q(0.535280085739468*pi,1.2473413121759003*pi) q[1];
U1q(0.152634840729795*pi,0.4363483222076998*pi) q[2];
U1q(0.339541701356641*pi,0.75350858036078*pi) q[3];
U1q(0.268178654249527*pi,1.7116053825796502*pi) q[4];
U1q(0.0823808728656546*pi,0.5901725730110301*pi) q[5];
U1q(0.870134134595413*pi,0.5776720371975399*pi) q[6];
U1q(0.27151133860358*pi,0.04591818450107743*pi) q[7];
U1q(0.37297541293805*pi,0.1897473117624342*pi) q[8];
U1q(0.796235907653938*pi,0.8618934723803315*pi) q[9];
U1q(0.302358994508822*pi,0.7662496075782306*pi) q[10];
U1q(0.679252256879562*pi,1.5231551860050743*pi) q[11];
U1q(0.907539940922818*pi,0.3410919745141898*pi) q[12];
U1q(0.68499628207061*pi,1.9146651246322528*pi) q[13];
U1q(0.353088756578738*pi,0.6680134223709868*pi) q[14];
U1q(0.326005480368563*pi,0.287219551771016*pi) q[15];
U1q(0.607819167044771*pi,1.675463673838721*pi) q[16];
U1q(0.637541265652583*pi,1.2694926297054998*pi) q[17];
U1q(0.521251740475494*pi,1.0536263760916391*pi) q[18];
U1q(0.356415950225726*pi,1.6857955483739802*pi) q[19];
U1q(0.671904983916695*pi,1.0189476360690464*pi) q[20];
U1q(0.350535812134207*pi,1.5473388005341482*pi) q[21];
U1q(0.692045646285638*pi,0.6555651635145585*pi) q[22];
U1q(0.899709797916109*pi,0.683948032392795*pi) q[23];
U1q(0.583906230639481*pi,1.103902532488927*pi) q[24];
U1q(0.748629919344845*pi,1.42309530759568*pi) q[25];
U1q(0.674005675914107*pi,1.6294930019410718*pi) q[26];
U1q(0.25174009278369*pi,0.4803734953673602*pi) q[27];
U1q(0.624109966154545*pi,1.034125668378832*pi) q[28];
U1q(0.666591858348307*pi,1.4671143939547502*pi) q[29];
U1q(0.714669684143831*pi,0.8922234358623*pi) q[30];
U1q(0.654720036364556*pi,0.549309349062165*pi) q[31];
U1q(0.378291612598324*pi,1.0430954085891502*pi) q[32];
U1q(0.583746508189646*pi,0.6894122704528101*pi) q[33];
U1q(0.609826728496522*pi,1.6637560307224837*pi) q[34];
U1q(0.464630438715498*pi,0.6114247973508999*pi) q[35];
U1q(0.516698871881246*pi,1.6653659804702898*pi) q[36];
U1q(0.716933504351416*pi,1.84565503745769*pi) q[37];
U1q(0.403947113412869*pi,0.05063786019152006*pi) q[38];
U1q(0.562827057719823*pi,0.8637521755889908*pi) q[39];
RZZ(0.5*pi) q[16],q[0];
RZZ(0.5*pi) q[23],q[1];
RZZ(0.5*pi) q[2],q[27];
RZZ(0.5*pi) q[3],q[17];
RZZ(0.5*pi) q[4],q[24];
RZZ(0.5*pi) q[39],q[5];
RZZ(0.5*pi) q[26],q[6];
RZZ(0.5*pi) q[7],q[19];
RZZ(0.5*pi) q[8],q[22];
RZZ(0.5*pi) q[25],q[9];
RZZ(0.5*pi) q[10],q[28];
RZZ(0.5*pi) q[11],q[12];
RZZ(0.5*pi) q[13],q[14];
RZZ(0.5*pi) q[37],q[15];
RZZ(0.5*pi) q[18],q[20];
RZZ(0.5*pi) q[32],q[21];
RZZ(0.5*pi) q[29],q[35];
RZZ(0.5*pi) q[30],q[34];
RZZ(0.5*pi) q[31],q[38];
RZZ(0.5*pi) q[33],q[36];
U1q(0.146607132562962*pi,1.39001989613078*pi) q[0];
U1q(0.609279850385494*pi,1.2361671962660603*pi) q[1];
U1q(0.48317377330224*pi,0.8790725306377896*pi) q[2];
U1q(0.703264667005752*pi,1.21394967696539*pi) q[3];
U1q(0.926066915765296*pi,0.90482825841866*pi) q[4];
U1q(0.585926708333307*pi,1.90897361933576*pi) q[5];
U1q(0.27257637577284*pi,1.8854881182547203*pi) q[6];
U1q(0.355662542643553*pi,0.9457315244825875*pi) q[7];
U1q(0.504102357002995*pi,0.1888693318929544*pi) q[8];
U1q(0.677900775726768*pi,0.6557094029929322*pi) q[9];
U1q(0.500644481331172*pi,0.04454130999866113*pi) q[10];
U1q(0.303300160402523*pi,0.4089074478265742*pi) q[11];
U1q(0.41727817852231*pi,0.6491484475555804*pi) q[12];
U1q(0.355006452782351*pi,0.549583372129173*pi) q[13];
U1q(0.426040468058333*pi,0.0593700622255966*pi) q[14];
U1q(0.703521499149412*pi,0.24607308407053008*pi) q[15];
U1q(0.731609291002215*pi,1.5660219710048615*pi) q[16];
U1q(0.53226343411691*pi,1.2396006510395896*pi) q[17];
U1q(0.384933602658021*pi,1.6385191567681883*pi) q[18];
U1q(0.83138558294345*pi,1.22830306207379*pi) q[19];
U1q(0.77324028484775*pi,0.2470912002070096*pi) q[20];
U1q(0.570943199336805*pi,1.8115272628045478*pi) q[21];
U1q(0.273852074156756*pi,0.6290992817374583*pi) q[22];
U1q(0.296915374947193*pi,0.34966764375408*pi) q[23];
U1q(0.31023094053591*pi,1.1844821070070002*pi) q[24];
U1q(0.382058046242536*pi,1.6022451800566104*pi) q[25];
U1q(0.271979112368311*pi,0.6051886918263922*pi) q[26];
U1q(0.579460655608335*pi,1.5956298809398701*pi) q[27];
U1q(0.449394109974023*pi,1.9601826610026198*pi) q[28];
U1q(0.329651782195395*pi,1.7226369652875206*pi) q[29];
U1q(0.531035420910483*pi,1.9651864202897502*pi) q[30];
U1q(0.227203766310956*pi,1.153872086496579*pi) q[31];
U1q(0.65502933864815*pi,1.0714453833031001*pi) q[32];
U1q(0.398767136431524*pi,0.9071338725058999*pi) q[33];
U1q(0.409712823565883*pi,0.44739793401220407*pi) q[34];
U1q(0.818309157223404*pi,0.004239821672199717*pi) q[35];
U1q(0.478796583691533*pi,1.4847329832959302*pi) q[36];
U1q(0.48641706340634*pi,0.24820541884590996*pi) q[37];
U1q(0.920857434105496*pi,0.17388729670207992*pi) q[38];
U1q(0.115111102448487*pi,0.7296942610988699*pi) q[39];
RZZ(0.5*pi) q[0],q[27];
RZZ(0.5*pi) q[1],q[35];
RZZ(0.5*pi) q[32],q[2];
RZZ(0.5*pi) q[14],q[3];
RZZ(0.5*pi) q[25],q[4];
RZZ(0.5*pi) q[5],q[21];
RZZ(0.5*pi) q[33],q[6];
RZZ(0.5*pi) q[7],q[37];
RZZ(0.5*pi) q[8],q[11];
RZZ(0.5*pi) q[38],q[9];
RZZ(0.5*pi) q[16],q[10];
RZZ(0.5*pi) q[15],q[12];
RZZ(0.5*pi) q[13],q[39];
RZZ(0.5*pi) q[17],q[20];
RZZ(0.5*pi) q[36],q[18];
RZZ(0.5*pi) q[23],q[19];
RZZ(0.5*pi) q[31],q[22];
RZZ(0.5*pi) q[34],q[24];
RZZ(0.5*pi) q[26],q[30];
RZZ(0.5*pi) q[29],q[28];
U1q(0.790699180782912*pi,0.9121327690285295*pi) q[0];
U1q(0.82616404313625*pi,0.4278089084363299*pi) q[1];
U1q(0.395726737555568*pi,1.4352578196485704*pi) q[2];
U1q(0.824094301602943*pi,0.10677379631756967*pi) q[3];
U1q(0.307632793360314*pi,1.1011432675437192*pi) q[4];
U1q(0.81211005746783*pi,0.35885177774651034*pi) q[5];
U1q(0.349051722447716*pi,0.4519996625122893*pi) q[6];
U1q(0.354122942258644*pi,0.024420492890246948*pi) q[7];
U1q(0.44053887140764*pi,1.5890711800237636*pi) q[8];
U1q(0.492796740515019*pi,0.9718459415646015*pi) q[9];
U1q(0.936032677464006*pi,1.0971726959822607*pi) q[10];
U1q(0.281858150638841*pi,1.8188012270891543*pi) q[11];
U1q(0.646430340979254*pi,0.038302532374509646*pi) q[12];
U1q(0.716787805506725*pi,1.8118976248503529*pi) q[13];
U1q(0.238629209521198*pi,1.675797599400596*pi) q[14];
U1q(0.021240612019142*pi,0.506592235187*pi) q[15];
U1q(0.516661006682247*pi,0.25580771238343125*pi) q[16];
U1q(0.84492405777598*pi,0.7094000759139298*pi) q[17];
U1q(0.867857523724668*pi,1.9029433633152895*pi) q[18];
U1q(0.251252309109101*pi,0.6241633359971104*pi) q[19];
U1q(0.668019272656495*pi,1.476200380937029*pi) q[20];
U1q(0.885503409604144*pi,0.3400790120471484*pi) q[21];
U1q(0.312476518492243*pi,1.2229259865120596*pi) q[22];
U1q(0.679720004584635*pi,1.2471239704916002*pi) q[23];
U1q(0.137440068066763*pi,1.6704211662133401*pi) q[24];
U1q(0.187391390645237*pi,1.1777037657987002*pi) q[25];
U1q(0.359170785310394*pi,1.6882290838963616*pi) q[26];
U1q(0.373409957046445*pi,1.57585367621442*pi) q[27];
U1q(0.443362607403099*pi,0.39566587128831987*pi) q[28];
U1q(0.261914499897932*pi,1.3373704524519994*pi) q[29];
U1q(0.40382417746299*pi,0.9548541583885601*pi) q[30];
U1q(0.346743326585917*pi,1.2067449410842892*pi) q[31];
U1q(0.397514933934881*pi,1.0109731001324906*pi) q[32];
U1q(0.396951520537667*pi,1.7079848625476304*pi) q[33];
U1q(0.446556630431693*pi,0.6215486496465346*pi) q[34];
U1q(0.577761015787666*pi,1.4352596983098698*pi) q[35];
U1q(0.561428936053922*pi,1.70577043439443*pi) q[36];
U1q(0.57344449758685*pi,1.9442155855238203*pi) q[37];
U1q(0.424865486107306*pi,1.3984828961234301*pi) q[38];
U1q(0.148517156899833*pi,0.5882918562381008*pi) q[39];
RZZ(0.5*pi) q[0],q[6];
RZZ(0.5*pi) q[1],q[34];
RZZ(0.5*pi) q[2],q[21];
RZZ(0.5*pi) q[37],q[3];
RZZ(0.5*pi) q[31],q[4];
RZZ(0.5*pi) q[7],q[5];
RZZ(0.5*pi) q[38],q[8];
RZZ(0.5*pi) q[9],q[12];
RZZ(0.5*pi) q[36],q[10];
RZZ(0.5*pi) q[39],q[11];
RZZ(0.5*pi) q[33],q[13];
RZZ(0.5*pi) q[14],q[28];
RZZ(0.5*pi) q[26],q[15];
RZZ(0.5*pi) q[16],q[29];
RZZ(0.5*pi) q[35],q[17];
RZZ(0.5*pi) q[18],q[24];
RZZ(0.5*pi) q[19],q[20];
RZZ(0.5*pi) q[30],q[22];
RZZ(0.5*pi) q[23],q[25];
RZZ(0.5*pi) q[32],q[27];
U1q(0.563771906344432*pi,0.651564384936*pi) q[0];
U1q(0.427368375995388*pi,1.8609335949766894*pi) q[1];
U1q(0.65331236782562*pi,0.3419481947333498*pi) q[2];
U1q(0.970454254421354*pi,0.53577885186575*pi) q[3];
U1q(0.657391732268575*pi,1.0620665082222995*pi) q[4];
U1q(0.816042977328725*pi,1.8074956863010705*pi) q[5];
U1q(0.503404019813899*pi,0.8038255743981999*pi) q[6];
U1q(0.244527226666307*pi,0.44226460649612775*pi) q[7];
U1q(0.566163522560117*pi,0.4827529089510829*pi) q[8];
U1q(0.249615013221583*pi,1.3196289963580217*pi) q[9];
U1q(0.486525244105805*pi,0.04071833366377042*pi) q[10];
U1q(0.437508005788064*pi,1.2870717794486746*pi) q[11];
U1q(0.506204838902347*pi,1.9302615414885*pi) q[12];
U1q(0.208880980434649*pi,0.9968180641537536*pi) q[13];
U1q(0.713336615553297*pi,1.6213301458091376*pi) q[14];
U1q(0.425046225317356*pi,0.9878821229897703*pi) q[15];
U1q(0.781832182318631*pi,0.38855965683105076*pi) q[16];
U1q(0.701758240742858*pi,1.3501655550541898*pi) q[17];
U1q(0.562077058062227*pi,0.295419542631409*pi) q[18];
U1q(0.638391411887319*pi,0.5363649341342995*pi) q[19];
U1q(0.265759676096699*pi,1.1329588283304997*pi) q[20];
U1q(0.262449504592619*pi,0.8111174077800776*pi) q[21];
U1q(0.357870422876287*pi,0.44644206823135946*pi) q[22];
U1q(0.285872447023178*pi,1.6426649694362805*pi) q[23];
U1q(0.47260813754314*pi,1.8193247688311907*pi) q[24];
U1q(0.352351416753605*pi,0.2786338082752202*pi) q[25];
U1q(0.244677455977213*pi,0.6699755137401713*pi) q[26];
U1q(0.729040640505276*pi,0.6483392798155698*pi) q[27];
U1q(0.513591170505726*pi,1.1727772955801496*pi) q[28];
U1q(0.670201360237271*pi,0.45006607689130007*pi) q[29];
U1q(0.626226678018816*pi,0.46044179261734985*pi) q[30];
U1q(0.442859586337219*pi,1.6390026025299385*pi) q[31];
U1q(0.363017565980453*pi,0.29911035046579926*pi) q[32];
U1q(0.598463295391843*pi,1.4430092327368609*pi) q[33];
U1q(0.47275174335698*pi,1.8250726079114248*pi) q[34];
U1q(0.601259277946329*pi,1.3985412233405707*pi) q[35];
U1q(0.410599869338172*pi,1.1915667195495008*pi) q[36];
U1q(0.323723825674315*pi,1.34549419337461*pi) q[37];
U1q(0.769738014643525*pi,1.8616777613465896*pi) q[38];
U1q(0.249789439609362*pi,0.18152409725049168*pi) q[39];
RZZ(0.5*pi) q[0],q[20];
RZZ(0.5*pi) q[33],q[1];
RZZ(0.5*pi) q[2],q[17];
RZZ(0.5*pi) q[8],q[3];
RZZ(0.5*pi) q[4],q[6];
RZZ(0.5*pi) q[30],q[5];
RZZ(0.5*pi) q[7],q[34];
RZZ(0.5*pi) q[9],q[24];
RZZ(0.5*pi) q[10],q[21];
RZZ(0.5*pi) q[31],q[11];
RZZ(0.5*pi) q[22],q[12];
RZZ(0.5*pi) q[13],q[18];
RZZ(0.5*pi) q[14],q[15];
RZZ(0.5*pi) q[36],q[16];
RZZ(0.5*pi) q[38],q[19];
RZZ(0.5*pi) q[23],q[39];
RZZ(0.5*pi) q[29],q[25];
RZZ(0.5*pi) q[26],q[27];
RZZ(0.5*pi) q[37],q[28];
RZZ(0.5*pi) q[32],q[35];
U1q(0.483849305729197*pi,0.6616069037049996*pi) q[0];
U1q(0.628017203568007*pi,1.4782702698822998*pi) q[1];
U1q(0.432589146100684*pi,1.8491924379103999*pi) q[2];
U1q(0.644523231879401*pi,1.3576644206676*pi) q[3];
U1q(0.318900980141213*pi,0.5259206959132996*pi) q[4];
U1q(0.681677870080626*pi,1.2290543872375999*pi) q[5];
U1q(0.643300115339535*pi,1.8458310017261006*pi) q[6];
U1q(0.754936435183775*pi,1.7781123382800281*pi) q[7];
U1q(0.384277934798644*pi,1.8294184447946833*pi) q[8];
U1q(0.32741556068306*pi,0.47304498906098047*pi) q[9];
U1q(0.568046240089678*pi,1.6460485091347703*pi) q[10];
U1q(0.457237420780049*pi,1.0316230253202647*pi) q[11];
U1q(0.241964934817444*pi,1.4881262940855997*pi) q[12];
U1q(0.042134383621418*pi,0.2380176791348525*pi) q[13];
U1q(0.385592697584933*pi,0.2967345678590174*pi) q[14];
U1q(0.540739533387557*pi,0.28784096137018*pi) q[15];
U1q(0.402353319575455*pi,0.6259909044887504*pi) q[16];
U1q(0.0488860006977406*pi,0.5773127893400805*pi) q[17];
U1q(0.619725193675479*pi,0.9438494430791398*pi) q[18];
U1q(0.358409949479206*pi,1.4062334786848005*pi) q[19];
U1q(0.775482350507945*pi,0.27650300223742974*pi) q[20];
U1q(0.502506113828029*pi,0.16749785894077895*pi) q[21];
U1q(0.533208231081566*pi,1.6770758134495587*pi) q[22];
U1q(0.483363111613297*pi,0.34362473590869946*pi) q[23];
U1q(0.532952374714844*pi,0.06186025998830935*pi) q[24];
U1q(0.600712719536087*pi,0.7772349291224003*pi) q[25];
U1q(0.460494520037702*pi,0.7459724030343313*pi) q[26];
U1q(0.546566329422048*pi,0.019573392025719727*pi) q[27];
U1q(0.674958492386512*pi,1.6301341497243502*pi) q[28];
U1q(0.587433011569233*pi,0.6316048137560006*pi) q[29];
U1q(0.505408964348577*pi,1.9459231318441805*pi) q[30];
U1q(0.391407389445536*pi,0.6910735907659991*pi) q[31];
U1q(0.61476855473336*pi,1.8672053181708996*pi) q[32];
U1q(0.743983624359672*pi,0.9583449093876109*pi) q[33];
U1q(0.516861496750147*pi,0.3340391328596244*pi) q[34];
U1q(0.104407838704309*pi,0.40466157077019993*pi) q[35];
U1q(0.523975861349965*pi,0.7888709994545007*pi) q[36];
U1q(0.300926559895116*pi,0.4883066036689696*pi) q[37];
U1q(0.702520892872661*pi,1.3900516095339999*pi) q[38];
U1q(0.683083679739451*pi,0.10187154196779069*pi) q[39];
rz(2.2059953060189006*pi) q[0];
rz(1.8935398206096892*pi) q[1];
rz(0.3748688654173993*pi) q[2];
rz(1.2190642297541991*pi) q[3];
rz(1.9604512600144002*pi) q[4];
rz(3.0644348484361004*pi) q[5];
rz(3.6604577077794005*pi) q[6];
rz(2.622659287184103*pi) q[7];
rz(2.711682822636117*pi) q[8];
rz(1.3158124783302192*pi) q[9];
rz(3.7728264383807293*pi) q[10];
rz(2.1266428284179746*pi) q[11];
rz(3.5289842377596*pi) q[12];
rz(1.3291952276180474*pi) q[13];
rz(2.1461312611432835*pi) q[14];
rz(3.0407325160226604*pi) q[15];
rz(2.4855156717911484*pi) q[16];
rz(1.7143613381023908*pi) q[17];
rz(3.1665500848371604*pi) q[18];
rz(3.049281362360899*pi) q[19];
rz(1.9993966724692704*pi) q[20];
rz(1.7773063071490824*pi) q[21];
rz(3.4496411210431397*pi) q[22];
rz(3.711625049611399*pi) q[23];
rz(1.9277873537874495*pi) q[24];
rz(2.8741777399089*pi) q[25];
rz(3.825757371077769*pi) q[26];
rz(2.7815018404283*pi) q[27];
rz(3.9116879262507*pi) q[28];
rz(1.8983669908290004*pi) q[29];
rz(0.4344059978323198*pi) q[30];
rz(3.005409445358401*pi) q[31];
rz(3.1268492769490006*pi) q[32];
rz(3.83496980593722*pi) q[33];
rz(3.1360104863996767*pi) q[34];
rz(0.8732157749180995*pi) q[35];
rz(1.409206603466*pi) q[36];
rz(0.30540250631847954*pi) q[37];
rz(3.6250383986598997*pi) q[38];
rz(3.6138414816190085*pi) q[39];
U1q(1.4838493057292*pi,1.867602209723926*pi) q[0];
U1q(3.628017203568007*pi,0.371810090491964*pi) q[1];
U1q(0.432589146100684*pi,1.224061303327767*pi) q[2];
U1q(0.644523231879401*pi,1.576728650421746*pi) q[3];
U1q(1.31890098014121*pi,1.486371955927696*pi) q[4];
U1q(1.68167787008063*pi,1.293489235673643*pi) q[5];
U1q(0.643300115339535*pi,0.506288709505434*pi) q[6];
U1q(0.754936435183775*pi,1.40077162546414*pi) q[7];
U1q(0.384277934798644*pi,1.541101267430824*pi) q[8];
U1q(3.32741556068306*pi,0.7888574673912501*pi) q[9];
U1q(0.568046240089678*pi,0.418874947515509*pi) q[10];
U1q(0.457237420780049*pi,0.158265853738286*pi) q[11];
U1q(1.24196493481744*pi,0.0171105318452115*pi) q[12];
U1q(0.042134383621418*pi,0.5672129067529099*pi) q[13];
U1q(1.38559269758493*pi,1.44286582900223*pi) q[14];
U1q(1.54073953338756*pi,0.328573477392838*pi) q[15];
U1q(1.40235331957545*pi,0.111506576279863*pi) q[16];
U1q(1.04888600069774*pi,1.29167412744247*pi) q[17];
U1q(0.619725193675479*pi,1.11039952791629*pi) q[18];
U1q(0.358409949479206*pi,1.455514841045734*pi) q[19];
U1q(0.775482350507945*pi,1.27589967470677*pi) q[20];
U1q(0.502506113828029*pi,0.944804166089856*pi) q[21];
U1q(0.533208231081566*pi,0.126716934492628*pi) q[22];
U1q(0.483363111613297*pi,1.055249785520152*pi) q[23];
U1q(0.532952374714844*pi,0.989647613775756*pi) q[24];
U1q(1.60071271953609*pi,0.651412669031338*pi) q[25];
U1q(0.460494520037702*pi,1.571729774112105*pi) q[26];
U1q(1.54656632942205*pi,1.801075232453971*pi) q[27];
U1q(0.674958492386512*pi,0.541822075975051*pi) q[28];
U1q(0.587433011569233*pi,1.529971804585022*pi) q[29];
U1q(3.505408964348578*pi,1.380329129676499*pi) q[30];
U1q(1.39140738944554*pi,0.6964830361243299*pi) q[31];
U1q(1.61476855473336*pi,1.9940545951199158*pi) q[32];
U1q(1.74398362435967*pi,1.793314715324828*pi) q[33];
U1q(0.516861496750147*pi,0.470049619259263*pi) q[34];
U1q(0.104407838704309*pi,0.277877345688335*pi) q[35];
U1q(0.523975861349965*pi,1.19807760292056*pi) q[36];
U1q(0.300926559895116*pi,1.793709109987451*pi) q[37];
U1q(0.702520892872661*pi,0.0150900081939356*pi) q[38];
U1q(1.68308367973945*pi,0.715713023586747*pi) q[39];
RZZ(0.5*pi) q[0],q[20];
RZZ(0.5*pi) q[33],q[1];
RZZ(0.5*pi) q[2],q[17];
RZZ(0.5*pi) q[8],q[3];
RZZ(0.5*pi) q[4],q[6];
RZZ(0.5*pi) q[30],q[5];
RZZ(0.5*pi) q[7],q[34];
RZZ(0.5*pi) q[9],q[24];
RZZ(0.5*pi) q[10],q[21];
RZZ(0.5*pi) q[31],q[11];
RZZ(0.5*pi) q[22],q[12];
RZZ(0.5*pi) q[13],q[18];
RZZ(0.5*pi) q[14],q[15];
RZZ(0.5*pi) q[36],q[16];
RZZ(0.5*pi) q[38],q[19];
RZZ(0.5*pi) q[23],q[39];
RZZ(0.5*pi) q[29],q[25];
RZZ(0.5*pi) q[26],q[27];
RZZ(0.5*pi) q[37],q[28];
RZZ(0.5*pi) q[32],q[35];
U1q(1.56377190634443*pi,1.8776447284929059*pi) q[0];
U1q(1.42736837599539*pi,0.9891467653975703*pi) q[1];
U1q(1.65331236782562*pi,1.7168170601507402*pi) q[2];
U1q(0.970454254421354*pi,1.7548430816199199*pi) q[3];
U1q(3.342608267731425*pi,1.9502261436186825*pi) q[4];
U1q(3.816042977328726*pi,1.7150479366101408*pi) q[5];
U1q(3.503404019813899*pi,1.464283282177584*pi) q[6];
U1q(0.244527226666307*pi,1.06492389368023*pi) q[7];
U1q(0.566163522560117*pi,1.1944357315872391*pi) q[8];
U1q(3.750384986778417*pi,0.9422734600942411*pi) q[9];
U1q(0.486525244105805*pi,1.8135447720445002*pi) q[10];
U1q(0.437508005788064*pi,0.4137146078667*pi) q[11];
U1q(1.50620483890235*pi,1.574975284442339*pi) q[12];
U1q(1.20888098043465*pi,0.3260132917717198*pi) q[13];
U1q(3.286663384446703*pi,0.11827025105206723*pi) q[14];
U1q(3.574953774682645*pi,1.6285323157732468*pi) q[15];
U1q(1.78183218231863*pi,0.348937823937514*pi) q[16];
U1q(1.70175824074286*pi,0.5188213617283575*pi) q[17];
U1q(0.562077058062227*pi,1.461969627468602*pi) q[18];
U1q(0.638391411887319*pi,1.5856462964952298*pi) q[19];
U1q(0.265759676096699*pi,0.132355500799803*pi) q[20];
U1q(1.26244950459262*pi,0.588423714929159*pi) q[21];
U1q(0.357870422876287*pi,0.8960831892744301*pi) q[22];
U1q(0.285872447023178*pi,0.35429001904773005*pi) q[23];
U1q(0.47260813754314*pi,0.747112122618639*pi) q[24];
U1q(3.647648583246395*pi,0.15001378987853098*pi) q[25];
U1q(0.244677455977213*pi,1.4957328848179001*pi) q[26];
U1q(3.270959359494724*pi,0.17230934466412554*pi) q[27];
U1q(0.513591170505726*pi,1.0844652218308561*pi) q[28];
U1q(1.67020136023727*pi,1.348433067720354*pi) q[29];
U1q(3.373773321981184*pi,0.8658104689033319*pi) q[30];
U1q(3.557140413662781*pi,0.7485540243603392*pi) q[31];
U1q(1.36301756598045*pi,1.5621495628250255*pi) q[32];
U1q(3.401536704608157*pi,1.3086503919755788*pi) q[33];
U1q(0.47275174335698*pi,0.96108309431107*pi) q[34];
U1q(0.601259277946329*pi,1.271756998258696*pi) q[35];
U1q(0.410599869338172*pi,1.6007733230155399*pi) q[36];
U1q(0.323723825674315*pi,1.65089669969309*pi) q[37];
U1q(0.769738014643525*pi,1.4867161600065102*pi) q[38];
U1q(3.750210560390638*pi,1.6360604683039819*pi) q[39];
RZZ(0.5*pi) q[0],q[6];
RZZ(0.5*pi) q[1],q[34];
RZZ(0.5*pi) q[2],q[21];
RZZ(0.5*pi) q[37],q[3];
RZZ(0.5*pi) q[31],q[4];
RZZ(0.5*pi) q[7],q[5];
RZZ(0.5*pi) q[38],q[8];
RZZ(0.5*pi) q[9],q[12];
RZZ(0.5*pi) q[36],q[10];
RZZ(0.5*pi) q[39],q[11];
RZZ(0.5*pi) q[33],q[13];
RZZ(0.5*pi) q[14],q[28];
RZZ(0.5*pi) q[26],q[15];
RZZ(0.5*pi) q[16],q[29];
RZZ(0.5*pi) q[35],q[17];
RZZ(0.5*pi) q[18],q[24];
RZZ(0.5*pi) q[19],q[20];
RZZ(0.5*pi) q[30],q[22];
RZZ(0.5*pi) q[23],q[25];
RZZ(0.5*pi) q[32],q[27];
U1q(1.79069918078291*pi,0.13821311258543392*pi) q[0];
U1q(0.82616404313625*pi,0.5560220788572061*pi) q[1];
U1q(3.604273262444432*pi,0.6235074352355232*pi) q[2];
U1q(1.82409430160294*pi,0.32583802607173995*pi) q[3];
U1q(3.3076327933603142*pi,1.911149384297239*pi) q[4];
U1q(1.81211005746783*pi,1.2664040280555808*pi) q[5];
U1q(3.650948277552284*pi,0.816109194063501*pi) q[6];
U1q(1.35412294225864*pi,1.647079780074353*pi) q[7];
U1q(1.44053887140764*pi,1.30075400265989*pi) q[8];
U1q(1.49279674051502*pi,0.2900565148876597*pi) q[9];
U1q(1.93603267746401*pi,1.8699991343630202*pi) q[10];
U1q(0.281858150638841*pi,0.9454440555071701*pi) q[11];
U1q(0.646430340979254*pi,1.6830162753283875*pi) q[12];
U1q(3.283212194493275*pi,1.5109337310750686*pi) q[13];
U1q(3.761370790478802*pi,0.06380279746060769*pi) q[14];
U1q(3.0212406120191417*pi,1.10982220357602*pi) q[15];
U1q(0.516661006682247*pi,0.21618587948986967*pi) q[16];
U1q(0.84492405777598*pi,0.8780558825880975*pi) q[17];
U1q(0.867857523724668*pi,0.06949344815248004*pi) q[18];
U1q(1.2512523091091*pi,0.6734446983579998*pi) q[19];
U1q(0.668019272656495*pi,1.4755970534063279*pi) q[20];
U1q(3.114496590395856*pi,1.059462110662097*pi) q[21];
U1q(3.312476518492243*pi,1.67256710755517*pi) q[22];
U1q(0.679720004584635*pi,1.95874902010305*pi) q[23];
U1q(0.137440068066763*pi,1.598208520000791*pi) q[24];
U1q(3.812608609354762*pi,1.250943832355049*pi) q[25];
U1q(1.35917078531039*pi,0.51398645497409*pi) q[26];
U1q(3.626590042953555*pi,1.244794948265263*pi) q[27];
U1q(0.443362607403099*pi,0.3073537975390299*pi) q[28];
U1q(3.738085500102068*pi,1.4611286921596172*pi) q[29];
U1q(3.596175822537009*pi,0.371398103132126*pi) q[30];
U1q(3.653256673414083*pi,0.18081168580599138*pi) q[31];
U1q(1.39751493393488*pi,0.27401231249169977*pi) q[32];
U1q(1.39695152053767*pi,1.043674762164808*pi) q[33];
U1q(0.446556630431693*pi,0.75755913604617*pi) q[34];
U1q(0.577761015787666*pi,0.30847547322799995*pi) q[35];
U1q(0.561428936053922*pi,1.1149770378604602*pi) q[36];
U1q(1.57344449758685*pi,1.2496180918422999*pi) q[37];
U1q(1.42486548610731*pi,0.023521294783349855*pi) q[38];
U1q(3.148517156899833*pi,0.2292927093164141*pi) q[39];
RZZ(0.5*pi) q[0],q[27];
RZZ(0.5*pi) q[1],q[35];
RZZ(0.5*pi) q[32],q[2];
RZZ(0.5*pi) q[14],q[3];
RZZ(0.5*pi) q[25],q[4];
RZZ(0.5*pi) q[5],q[21];
RZZ(0.5*pi) q[33],q[6];
RZZ(0.5*pi) q[7],q[37];
RZZ(0.5*pi) q[8],q[11];
RZZ(0.5*pi) q[38],q[9];
RZZ(0.5*pi) q[16],q[10];
RZZ(0.5*pi) q[15],q[12];
RZZ(0.5*pi) q[13],q[39];
RZZ(0.5*pi) q[17],q[20];
RZZ(0.5*pi) q[36],q[18];
RZZ(0.5*pi) q[23],q[19];
RZZ(0.5*pi) q[31],q[22];
RZZ(0.5*pi) q[34],q[24];
RZZ(0.5*pi) q[26],q[30];
RZZ(0.5*pi) q[29],q[28];
U1q(3.146607132562962*pi,0.6603259854831833*pi) q[0];
U1q(0.609279850385494*pi,0.3643803666869365*pi) q[1];
U1q(1.48317377330224*pi,1.1796927242463031*pi) q[2];
U1q(3.296735332994248*pi,1.2186621454239166*pi) q[3];
U1q(0.926066915765296*pi,1.714834375172169*pi) q[4];
U1q(3.414073291666693*pi,1.7162821864663185*pi) q[5];
U1q(3.72742362422716*pi,1.382620738321071*pi) q[6];
U1q(3.644337457356447*pi,0.7257687484820132*pi) q[7];
U1q(3.4958976429970052*pi,1.7009558507906979*pi) q[8];
U1q(0.677900775726768*pi,1.9739199763159903*pi) q[9];
U1q(3.499355518668828*pi,1.92263052034661*pi) q[10];
U1q(1.30330016040252*pi,1.5355502762445896*pi) q[11];
U1q(1.41727817852231*pi,1.2938621905094676*pi) q[12];
U1q(1.35500645278235*pi,1.7732479837962538*pi) q[13];
U1q(1.42604046805833*pi,1.6802303346356178*pi) q[14];
U1q(0.703521499149412*pi,1.84930305245956*pi) q[15];
U1q(1.73160929100222*pi,1.526400138111301*pi) q[16];
U1q(1.53226343411691*pi,0.4082564577137573*pi) q[17];
U1q(1.38493360265802*pi,0.80506924160537*pi) q[18];
U1q(3.16861441705655*pi,1.069304972281314*pi) q[19];
U1q(0.77324028484775*pi,1.24648787267631*pi) q[20];
U1q(3.429056800663195*pi,0.5880138599046977*pi) q[21];
U1q(3.726147925843244*pi,0.26639381232977666*pi) q[22];
U1q(1.29691537494719*pi,1.0612926933655302*pi) q[23];
U1q(0.31023094053591*pi,0.1122694607944501*pi) q[24];
U1q(3.617941953757464*pi,0.826402418097145*pi) q[25];
U1q(1.27197911236831*pi,0.5970268470440576*pi) q[26];
U1q(3.420539344391664*pi,0.2250187435398261*pi) q[27];
U1q(1.44939410997402*pi,1.8718705872533299*pi) q[28];
U1q(1.3296517821954*pi,0.07586217932414074*pi) q[29];
U1q(1.53103542091048*pi,1.3610658412309324*pi) q[30];
U1q(1.22720376631096*pi,0.2336845403937038*pi) q[31];
U1q(3.3449706613518497*pi,0.21354002932108962*pi) q[32];
U1q(3.398767136431524*pi,0.2428237721230695*pi) q[33];
U1q(0.409712823565883*pi,1.58340842041184*pi) q[34];
U1q(0.818309157223404*pi,1.8774555965903197*pi) q[35];
U1q(0.478796583691533*pi,1.8939395867619604*pi) q[36];
U1q(3.51358293659366*pi,0.9456282585202072*pi) q[37];
U1q(1.9208574341055*pi,1.2481168942046947*pi) q[38];
U1q(1.11511110244849*pi,0.37069511417718415*pi) q[39];
RZZ(0.5*pi) q[16],q[0];
RZZ(0.5*pi) q[23],q[1];
RZZ(0.5*pi) q[2],q[27];
RZZ(0.5*pi) q[3],q[17];
RZZ(0.5*pi) q[4],q[24];
RZZ(0.5*pi) q[39],q[5];
RZZ(0.5*pi) q[26],q[6];
RZZ(0.5*pi) q[7],q[19];
RZZ(0.5*pi) q[8],q[22];
RZZ(0.5*pi) q[25],q[9];
RZZ(0.5*pi) q[10],q[28];
RZZ(0.5*pi) q[11],q[12];
RZZ(0.5*pi) q[13],q[14];
RZZ(0.5*pi) q[37],q[15];
RZZ(0.5*pi) q[18],q[20];
RZZ(0.5*pi) q[32],q[21];
RZZ(0.5*pi) q[29],q[35];
RZZ(0.5*pi) q[30],q[34];
RZZ(0.5*pi) q[31],q[38];
RZZ(0.5*pi) q[33],q[36];
U1q(1.50256905072322*pi,0.2026785620834084*pi) q[0];
U1q(1.53528008573947*pi,0.37555448259677604*pi) q[1];
U1q(1.15263484072979*pi,0.736968515816212*pi) q[2];
U1q(1.33954170135664*pi,1.6791032420285248*pi) q[3];
U1q(1.26817865424953*pi,1.5216114993331695*pi) q[4];
U1q(3.9176191271343446*pi,0.03508323279104886*pi) q[5];
U1q(3.1298658654045868*pi,0.6904368193782506*pi) q[6];
U1q(1.27151133860358*pi,1.625582088463525*pi) q[7];
U1q(3.62702458706195*pi,0.7000778709212172*pi) q[8];
U1q(1.79623590765394*pi,0.18010404570340022*pi) q[9];
U1q(1.30235899450882*pi,1.2009222227670433*pi) q[10];
U1q(1.67925225687956*pi,1.4213025380660858*pi) q[11];
U1q(3.092460059077183*pi,1.6019186635508529*pi) q[12];
U1q(1.68499628207061*pi,0.13832973629934298*pi) q[13];
U1q(0.353088756578738*pi,1.2888736947810076*pi) q[14];
U1q(0.326005480368563*pi,1.89044952016004*pi) q[15];
U1q(1.60781916704477*pi,1.4169584352774418*pi) q[16];
U1q(3.362458734347416*pi,0.37836447904784265*pi) q[17];
U1q(1.52125174047549*pi,1.3899620222819102*pi) q[18];
U1q(3.643584049774274*pi,1.6118124859811234*pi) q[19];
U1q(1.6719049839167*pi,0.018344308538349896*pi) q[20];
U1q(3.649464187865793*pi,1.8522023221750938*pi) q[21];
U1q(3.307954353714362*pi,1.2399279305526765*pi) q[22];
U1q(3.100290202083891*pi,1.7270123047268173*pi) q[23];
U1q(1.58390623063948*pi,1.03168988627638*pi) q[24];
U1q(3.251370080655155*pi,1.0055522905580654*pi) q[25];
U1q(0.674005675914107*pi,1.6213311571587283*pi) q[26];
U1q(1.25174009278369*pi,0.3402751291123358*pi) q[27];
U1q(1.62410996615454*pi,1.7979275798771202*pi) q[28];
U1q(0.666591858348307*pi,0.8203396079913707*pi) q[29];
U1q(0.714669684143831*pi,1.2881028568034827*pi) q[30];
U1q(0.654720036364556*pi,0.6291218029592938*pi) q[31];
U1q(1.37829161259832*pi,0.24189000403504712*pi) q[32];
U1q(1.58374650818965*pi,0.4605453741761574*pi) q[33];
U1q(0.609826728496522*pi,1.79976651712213*pi) q[34];
U1q(1.4646304387155*pi,1.4846405722690204*pi) q[35];
U1q(0.516698871881246*pi,0.0745725839363196*pi) q[36];
U1q(1.71693350435142*pi,0.34817863990842746*pi) q[37];
U1q(1.40394711341287*pi,0.1248674576941351*pi) q[38];
U1q(1.56282705771982*pi,1.2366371996870589*pi) q[39];
RZZ(0.5*pi) q[0],q[3];
RZZ(0.5*pi) q[1],q[21];
RZZ(0.5*pi) q[2],q[20];
RZZ(0.5*pi) q[22],q[4];
RZZ(0.5*pi) q[5],q[24];
RZZ(0.5*pi) q[36],q[6];
RZZ(0.5*pi) q[7],q[15];
RZZ(0.5*pi) q[19],q[8];
RZZ(0.5*pi) q[29],q[9];
RZZ(0.5*pi) q[10],q[37];
RZZ(0.5*pi) q[11],q[28];
RZZ(0.5*pi) q[23],q[12];
RZZ(0.5*pi) q[13],q[31];
RZZ(0.5*pi) q[38],q[14];
RZZ(0.5*pi) q[16],q[25];
RZZ(0.5*pi) q[17],q[34];
RZZ(0.5*pi) q[26],q[18];
RZZ(0.5*pi) q[33],q[27];
RZZ(0.5*pi) q[30],q[35];
RZZ(0.5*pi) q[32],q[39];
U1q(1.32273842513657*pi,1.2906396948519085*pi) q[0];
U1q(1.22740556140202*pi,1.656487749620867*pi) q[1];
U1q(1.76385358182292*pi,1.734158085478212*pi) q[2];
U1q(0.42411318810458*pi,0.46030135837626496*pi) q[3];
U1q(3.328155142044111*pi,1.4309935533568092*pi) q[4];
U1q(1.72361634180954*pi,0.5296036963768382*pi) q[5];
U1q(1.45887201315778*pi,0.05196663654981659*pi) q[6];
U1q(0.392592596792175*pi,1.6365677328622352*pi) q[7];
U1q(1.78873016279253*pi,0.26614573184776535*pi) q[8];
U1q(1.87712504787906*pi,0.7833972268871525*pi) q[9];
U1q(0.475196980526434*pi,0.6245148364509934*pi) q[10];
U1q(0.0610862639565709*pi,1.8049688442658365*pi) q[11];
U1q(1.28498234057837*pi,1.5643631554241608*pi) q[12];
U1q(1.43161948830389*pi,1.0606867545862473*pi) q[13];
U1q(0.174529702659025*pi,1.887712696676238*pi) q[14];
U1q(0.482898704837826*pi,1.33788042631577*pi) q[15];
U1q(0.394141747398708*pi,1.0281794330791718*pi) q[16];
U1q(1.19725749918639*pi,0.47394509645573457*pi) q[17];
U1q(0.724898672545206*pi,1.4347033695158107*pi) q[18];
U1q(1.23388281781834*pi,1.4915817536721416*pi) q[19];
U1q(1.4670463418874*pi,1.3402536965386664*pi) q[20];
U1q(3.35728790354968*pi,1.9849120204711257*pi) q[21];
U1q(1.65373596096344*pi,0.6219443588777738*pi) q[22];
U1q(1.92239426845415*pi,1.1598964489078742*pi) q[23];
U1q(1.22400170257014*pi,0.10259564021182266*pi) q[24];
U1q(1.55410976818524*pi,0.11669500397133437*pi) q[25];
U1q(0.915885432518273*pi,0.7165901004514881*pi) q[26];
U1q(0.422987213525969*pi,1.0703590238874887*pi) q[27];
U1q(0.366933603614469*pi,1.0334098582519902*pi) q[28];
U1q(0.733396884291223*pi,0.6752378955688609*pi) q[29];
U1q(0.535692735308016*pi,0.7483743828908724*pi) q[30];
U1q(0.731190102397018*pi,1.546474936624834*pi) q[31];
U1q(0.172945511628788*pi,0.9714099274797672*pi) q[32];
U1q(0.586273835806525*pi,0.6621468871577383*pi) q[33];
U1q(0.216420951432601*pi,0.96195485871247*pi) q[34];
U1q(1.71912286042942*pi,0.7547719833307971*pi) q[35];
U1q(0.210300563758592*pi,0.4870935663453295*pi) q[36];
U1q(0.889295208161124*pi,1.2968667398764282*pi) q[37];
U1q(3.825921595027834*pi,1.3966055110492137*pi) q[38];
U1q(0.807288534126907*pi,1.7029106859920549*pi) q[39];
rz(2.7093603051480915*pi) q[0];
rz(2.343512250379133*pi) q[1];
rz(2.265841914521788*pi) q[2];
rz(3.539698641623735*pi) q[3];
rz(0.5690064466431908*pi) q[4];
rz(3.4703963036231618*pi) q[5];
rz(3.9480333634501834*pi) q[6];
rz(2.3634322671377648*pi) q[7];
rz(1.7338542681522346*pi) q[8];
rz(1.2166027731128475*pi) q[9];
rz(3.3754851635490066*pi) q[10];
rz(0.19503115573416352*pi) q[11];
rz(0.4356368445758392*pi) q[12];
rz(0.9393132454137527*pi) q[13];
rz(2.112287303323762*pi) q[14];
rz(2.66211957368423*pi) q[15];
rz(0.9718205669208281*pi) q[16];
rz(3.5260549035442654*pi) q[17];
rz(2.5652966304841893*pi) q[18];
rz(2.5084182463278584*pi) q[19];
rz(2.6597463034613336*pi) q[20];
rz(2.0150879795288743*pi) q[21];
rz(1.3780556411222262*pi) q[22];
rz(0.8401035510921258*pi) q[23];
rz(3.8974043597881773*pi) q[24];
rz(3.8833049960286656*pi) q[25];
rz(1.283409899548512*pi) q[26];
rz(0.9296409761125112*pi) q[27];
rz(2.96659014174801*pi) q[28];
rz(1.324762104431139*pi) q[29];
rz(1.2516256171091276*pi) q[30];
rz(2.453525063375166*pi) q[31];
rz(1.0285900725202328*pi) q[32];
rz(1.3378531128422617*pi) q[33];
rz(3.03804514128753*pi) q[34];
rz(3.245228016669203*pi) q[35];
rz(1.5129064336546705*pi) q[36];
rz(0.7031332601235718*pi) q[37];
rz(0.6033944889507863*pi) q[38];
rz(0.29708931400794514*pi) q[39];
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
