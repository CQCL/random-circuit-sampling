OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.672309370181571*pi,0.901283446962215*pi) q[0];
U1q(0.918082612592165*pi,0.407322070509609*pi) q[1];
U1q(0.324474994180891*pi,1.159010361616764*pi) q[2];
U1q(0.535672123220657*pi,1.643832825253984*pi) q[3];
U1q(0.254302814297497*pi,0.790563784738303*pi) q[4];
U1q(0.573992734592314*pi,0.835374596657327*pi) q[5];
U1q(0.856003246301598*pi,1.822093364013816*pi) q[6];
U1q(0.588409893280998*pi,0.661467340555884*pi) q[7];
U1q(0.948970552033977*pi,1.05859438680477*pi) q[8];
U1q(0.609411984331565*pi,0.732740320662187*pi) q[9];
U1q(0.736767479739608*pi,0.37295365126154*pi) q[10];
U1q(0.285107444945941*pi,1.9233780852226428*pi) q[11];
U1q(0.185066656451908*pi,1.4960795226657368*pi) q[12];
U1q(0.442215948013406*pi,0.288008646678078*pi) q[13];
U1q(0.488182481580334*pi,1.7188215989952629*pi) q[14];
U1q(0.256123521574214*pi,0.258599845299442*pi) q[15];
U1q(0.842312062049888*pi,0.352762323528094*pi) q[16];
U1q(0.933424583072536*pi,0.867386466899169*pi) q[17];
U1q(0.192523782626516*pi,0.109403757118756*pi) q[18];
U1q(0.402080597442105*pi,1.2731000190509532*pi) q[19];
U1q(0.204572297755441*pi,0.0126174423225527*pi) q[20];
U1q(0.633890975153144*pi,1.2180424245719519*pi) q[21];
U1q(0.397064739829497*pi,1.88804442528573*pi) q[22];
U1q(0.2575203110677*pi,0.0445171979844414*pi) q[23];
U1q(0.943881129140043*pi,0.421380264517593*pi) q[24];
U1q(0.18701332468102*pi,0.309364151525371*pi) q[25];
U1q(0.5593668235434*pi,1.2264706438993351*pi) q[26];
U1q(0.558741823618952*pi,1.640982941314658*pi) q[27];
U1q(0.829443150293908*pi,0.632240319839256*pi) q[28];
U1q(0.873701964408399*pi,0.216959314667747*pi) q[29];
U1q(0.591430984748715*pi,1.13709314332106*pi) q[30];
U1q(0.428935864160635*pi,1.4228179264226148*pi) q[31];
U1q(0.64267914659924*pi,0.276772252210736*pi) q[32];
U1q(0.786691504758194*pi,1.288051496408041*pi) q[33];
U1q(0.839450113790494*pi,1.6931228732413501*pi) q[34];
U1q(0.698546254200801*pi,1.39075788081485*pi) q[35];
U1q(0.723704981298877*pi,1.9760655185754705*pi) q[36];
U1q(0.547046051530567*pi,0.986846373166609*pi) q[37];
U1q(0.495947291342207*pi,0.0633851472599204*pi) q[38];
U1q(0.175604743184798*pi,1.859847283944575*pi) q[39];
RZZ(0.5*pi) q[28],q[0];
RZZ(0.5*pi) q[1],q[14];
RZZ(0.5*pi) q[2],q[10];
RZZ(0.5*pi) q[3],q[11];
RZZ(0.5*pi) q[37],q[4];
RZZ(0.5*pi) q[5],q[29];
RZZ(0.5*pi) q[17],q[6];
RZZ(0.5*pi) q[7],q[32];
RZZ(0.5*pi) q[8],q[38];
RZZ(0.5*pi) q[9],q[13];
RZZ(0.5*pi) q[22],q[12];
RZZ(0.5*pi) q[33],q[15];
RZZ(0.5*pi) q[16],q[23];
RZZ(0.5*pi) q[24],q[18];
RZZ(0.5*pi) q[20],q[19];
RZZ(0.5*pi) q[21],q[25];
RZZ(0.5*pi) q[26],q[39];
RZZ(0.5*pi) q[27],q[30];
RZZ(0.5*pi) q[31],q[35];
RZZ(0.5*pi) q[36],q[34];
U1q(0.213827826088767*pi,1.6650075261697301*pi) q[0];
U1q(0.654246259164061*pi,1.767949924459787*pi) q[1];
U1q(0.355958158515112*pi,0.6412355364862399*pi) q[2];
U1q(0.487401325066486*pi,0.06291072925042007*pi) q[3];
U1q(0.485265707375159*pi,1.1404059292865698*pi) q[4];
U1q(0.0618859542230729*pi,1.132966620994892*pi) q[5];
U1q(0.346556831709785*pi,0.44470673543400996*pi) q[6];
U1q(0.796297003438818*pi,0.8787017759976301*pi) q[7];
U1q(0.556724741827128*pi,0.456580962983026*pi) q[8];
U1q(0.611022646414029*pi,1.860259386064126*pi) q[9];
U1q(0.657811393119923*pi,0.10311370451744994*pi) q[10];
U1q(0.144623845588828*pi,1.80639017964879*pi) q[11];
U1q(0.759550953422396*pi,1.32788887192355*pi) q[12];
U1q(0.555375017580017*pi,1.8511459707336102*pi) q[13];
U1q(0.845880858494237*pi,1.9784876478125302*pi) q[14];
U1q(0.423793662497201*pi,1.9798343817418602*pi) q[15];
U1q(0.497115637361786*pi,1.63401480395237*pi) q[16];
U1q(0.27529351728069*pi,0.0614174863031292*pi) q[17];
U1q(0.637744695322297*pi,0.020773499964360065*pi) q[18];
U1q(0.677236350454037*pi,1.9114028473285298*pi) q[19];
U1q(0.429364593239561*pi,0.46386079956289006*pi) q[20];
U1q(0.751452941795359*pi,1.8657359233471098*pi) q[21];
U1q(0.676077107962804*pi,0.9110565309743799*pi) q[22];
U1q(0.517148081603898*pi,1.4690758746765402*pi) q[23];
U1q(0.916895768145575*pi,0.9788278354739299*pi) q[24];
U1q(0.235517926652173*pi,0.9428309844073*pi) q[25];
U1q(0.746285813818159*pi,1.1772767760374299*pi) q[26];
U1q(0.204833659297162*pi,0.10844230058805016*pi) q[27];
U1q(0.52339192570895*pi,0.8263985659973601*pi) q[28];
U1q(0.778968412903949*pi,1.73033188543496*pi) q[29];
U1q(0.663581753382743*pi,1.547428799717041*pi) q[30];
U1q(0.864042071944683*pi,0.3373539096871001*pi) q[31];
U1q(0.894648872953333*pi,1.97613840481102*pi) q[32];
U1q(0.760298349246581*pi,0.42837443494458016*pi) q[33];
U1q(0.435489238895306*pi,1.6565148857080398*pi) q[34];
U1q(0.622954707475547*pi,1.172515299887918*pi) q[35];
U1q(0.338698101430591*pi,1.3738638704410704*pi) q[36];
U1q(0.175282837392605*pi,1.257169292621539*pi) q[37];
U1q(0.398148408219994*pi,0.5041195013725099*pi) q[38];
U1q(0.263536514464754*pi,1.77940609463956*pi) q[39];
RZZ(0.5*pi) q[8],q[0];
RZZ(0.5*pi) q[1],q[6];
RZZ(0.5*pi) q[23],q[2];
RZZ(0.5*pi) q[3],q[25];
RZZ(0.5*pi) q[20],q[4];
RZZ(0.5*pi) q[35],q[5];
RZZ(0.5*pi) q[7],q[24];
RZZ(0.5*pi) q[9],q[19];
RZZ(0.5*pi) q[39],q[10];
RZZ(0.5*pi) q[18],q[11];
RZZ(0.5*pi) q[37],q[12];
RZZ(0.5*pi) q[13],q[29];
RZZ(0.5*pi) q[31],q[14];
RZZ(0.5*pi) q[28],q[15];
RZZ(0.5*pi) q[16],q[27];
RZZ(0.5*pi) q[17],q[38];
RZZ(0.5*pi) q[32],q[21];
RZZ(0.5*pi) q[33],q[22];
RZZ(0.5*pi) q[26],q[34];
RZZ(0.5*pi) q[36],q[30];
U1q(0.658030199625267*pi,1.51962161271187*pi) q[0];
U1q(0.831093638472686*pi,0.5708731298071799*pi) q[1];
U1q(0.391150704971497*pi,0.9772666772552299*pi) q[2];
U1q(0.546521276358347*pi,1.8149040778824697*pi) q[3];
U1q(0.67161743332876*pi,1.77918981470454*pi) q[4];
U1q(0.441095672550655*pi,0.04678309037322004*pi) q[5];
U1q(0.236233783963789*pi,1.00303691299687*pi) q[6];
U1q(0.176540823576226*pi,0.7941133823074802*pi) q[7];
U1q(0.355978328412678*pi,0.024814544573730002*pi) q[8];
U1q(0.441156212699497*pi,1.3620614710442398*pi) q[9];
U1q(0.128061423031934*pi,1.7454951782499304*pi) q[10];
U1q(0.737433986498467*pi,0.08762759727420022*pi) q[11];
U1q(0.607524295877026*pi,0.91566476299678*pi) q[12];
U1q(0.561361182062328*pi,1.7310638324370498*pi) q[13];
U1q(0.284774561392465*pi,1.2851687937503202*pi) q[14];
U1q(0.706056268631412*pi,0.12826402117832014*pi) q[15];
U1q(0.253977086519818*pi,1.6947983490057403*pi) q[16];
U1q(0.230385951779707*pi,0.23399344095602004*pi) q[17];
U1q(0.127372676821537*pi,0.2448272516264498*pi) q[18];
U1q(0.461053711401456*pi,0.0235207357020899*pi) q[19];
U1q(0.166931471891192*pi,0.23460963080226005*pi) q[20];
U1q(0.81301518007862*pi,0.8004647953413597*pi) q[21];
U1q(0.275546824613957*pi,1.30146365301096*pi) q[22];
U1q(0.655864678949939*pi,1.9382172961557602*pi) q[23];
U1q(0.883012228436997*pi,1.9999940765641702*pi) q[24];
U1q(0.742716983800467*pi,0.06821258391055984*pi) q[25];
U1q(0.350753196213649*pi,1.41400457499227*pi) q[26];
U1q(0.80814970845417*pi,1.85857686619046*pi) q[27];
U1q(0.168365650192537*pi,1.72294475968076*pi) q[28];
U1q(0.293373535266162*pi,0.8109184726346701*pi) q[29];
U1q(0.811723674058604*pi,1.6535129355942102*pi) q[30];
U1q(0.0744526989063991*pi,0.24056513768380006*pi) q[31];
U1q(0.873228849810267*pi,0.6720407380555198*pi) q[32];
U1q(0.500153670269019*pi,1.1498582776488098*pi) q[33];
U1q(0.236284323999378*pi,0.08840408924976018*pi) q[34];
U1q(0.319423124101668*pi,1.5552753915362096*pi) q[35];
U1q(0.2059025218493*pi,0.16304395247976977*pi) q[36];
U1q(0.662222145580918*pi,1.8264440933701498*pi) q[37];
U1q(0.39365502321315*pi,0.3422423789915201*pi) q[38];
U1q(0.396056666275*pi,0.1167389529365499*pi) q[39];
RZZ(0.5*pi) q[21],q[0];
RZZ(0.5*pi) q[1],q[12];
RZZ(0.5*pi) q[30],q[2];
RZZ(0.5*pi) q[3],q[23];
RZZ(0.5*pi) q[6],q[4];
RZZ(0.5*pi) q[36],q[5];
RZZ(0.5*pi) q[7],q[28];
RZZ(0.5*pi) q[16],q[8];
RZZ(0.5*pi) q[37],q[9];
RZZ(0.5*pi) q[25],q[10];
RZZ(0.5*pi) q[35],q[11];
RZZ(0.5*pi) q[22],q[13];
RZZ(0.5*pi) q[14],q[38];
RZZ(0.5*pi) q[18],q[15];
RZZ(0.5*pi) q[17],q[19];
RZZ(0.5*pi) q[20],q[32];
RZZ(0.5*pi) q[39],q[24];
RZZ(0.5*pi) q[26],q[29];
RZZ(0.5*pi) q[31],q[27];
RZZ(0.5*pi) q[33],q[34];
U1q(0.695663965456627*pi,0.10726801708812017*pi) q[0];
U1q(0.852116006667019*pi,1.6660385160280002*pi) q[1];
U1q(0.609656622757743*pi,0.9951879051063299*pi) q[2];
U1q(0.757549343407354*pi,1.4023538043003203*pi) q[3];
U1q(0.833661203696941*pi,1.6955592563404096*pi) q[4];
U1q(0.197390117695739*pi,0.8027839936301904*pi) q[5];
U1q(0.758566509544601*pi,1.5326036262993004*pi) q[6];
U1q(0.0798473877199773*pi,0.11590056828650042*pi) q[7];
U1q(0.704965715893424*pi,1.2668890941251796*pi) q[8];
U1q(0.331979944133116*pi,1.9742506954895003*pi) q[9];
U1q(0.640074270117702*pi,1.3355250464828696*pi) q[10];
U1q(0.686252620732444*pi,1.7708665435712394*pi) q[11];
U1q(0.617573283043262*pi,0.49272385718407996*pi) q[12];
U1q(0.473332712844794*pi,1.73772904735723*pi) q[13];
U1q(0.899522841730277*pi,1.6710007504510003*pi) q[14];
U1q(0.459728412674538*pi,1.0287540131849502*pi) q[15];
U1q(0.637691390017962*pi,1.03986561808642*pi) q[16];
U1q(0.75264505036672*pi,1.7607704064487004*pi) q[17];
U1q(0.717644937729767*pi,1.58100421966218*pi) q[18];
U1q(0.337906138453519*pi,1.2453272151731998*pi) q[19];
U1q(0.734429350722258*pi,1.3077115235034498*pi) q[20];
U1q(0.21134986593103*pi,0.0778958368743794*pi) q[21];
U1q(0.351815645703007*pi,1.3188782121225096*pi) q[22];
U1q(0.156227662412081*pi,0.30223546781343025*pi) q[23];
U1q(0.633115562559979*pi,1.7533853968830808*pi) q[24];
U1q(0.894872769389916*pi,0.5987025159863499*pi) q[25];
U1q(0.348519276958677*pi,0.6282688926762896*pi) q[26];
U1q(0.610290008739048*pi,1.6769794153099706*pi) q[27];
U1q(0.463805272587296*pi,1.7248982582321393*pi) q[28];
U1q(0.577259416111694*pi,0.9048698399078399*pi) q[29];
U1q(0.184955943162868*pi,0.4150882885997502*pi) q[30];
U1q(0.706326913625687*pi,0.9059165532326903*pi) q[31];
U1q(0.675305883709786*pi,0.3137070924737104*pi) q[32];
U1q(0.539430636440376*pi,1.2483354954689592*pi) q[33];
U1q(0.0958783942970741*pi,1.4099896245456902*pi) q[34];
U1q(0.194291551281985*pi,0.37633893708528987*pi) q[35];
U1q(0.263858812861597*pi,1.10795657100895*pi) q[36];
U1q(0.687712737331433*pi,0.18315647521699008*pi) q[37];
U1q(0.750947983972528*pi,0.8618820267671996*pi) q[38];
U1q(0.421186401190845*pi,1.1439191689526202*pi) q[39];
RZZ(0.5*pi) q[24],q[0];
RZZ(0.5*pi) q[1],q[31];
RZZ(0.5*pi) q[12],q[2];
RZZ(0.5*pi) q[3],q[4];
RZZ(0.5*pi) q[5],q[23];
RZZ(0.5*pi) q[20],q[6];
RZZ(0.5*pi) q[7],q[17];
RZZ(0.5*pi) q[27],q[8];
RZZ(0.5*pi) q[9],q[30];
RZZ(0.5*pi) q[19],q[10];
RZZ(0.5*pi) q[11],q[13];
RZZ(0.5*pi) q[14],q[25];
RZZ(0.5*pi) q[16],q[15];
RZZ(0.5*pi) q[22],q[18];
RZZ(0.5*pi) q[35],q[21];
RZZ(0.5*pi) q[26],q[32];
RZZ(0.5*pi) q[28],q[34];
RZZ(0.5*pi) q[37],q[29];
RZZ(0.5*pi) q[33],q[36];
RZZ(0.5*pi) q[39],q[38];
U1q(0.499550077147573*pi,1.9103481974179193*pi) q[0];
U1q(0.364678909755776*pi,1.0737822218744792*pi) q[1];
U1q(0.718604450111276*pi,0.5256078458845304*pi) q[2];
U1q(0.473190363174292*pi,1.4604627533057002*pi) q[3];
U1q(0.316262186568352*pi,1.26378326008808*pi) q[4];
U1q(0.925786564210907*pi,1.0318104170085096*pi) q[5];
U1q(0.599502299098519*pi,1.7961565578776693*pi) q[6];
U1q(0.205088165573879*pi,0.38247110410113017*pi) q[7];
U1q(0.693487249219829*pi,0.2363598404517404*pi) q[8];
U1q(0.539272593586034*pi,1.1288895588643406*pi) q[9];
U1q(0.713343928185911*pi,1.4117672346605001*pi) q[10];
U1q(0.518742180169482*pi,0.5305721526937806*pi) q[11];
U1q(0.650936873645741*pi,1.9085174727375094*pi) q[12];
U1q(0.221241516064147*pi,1.2594316504388203*pi) q[13];
U1q(0.0387791486436207*pi,1.6708654866404107*pi) q[14];
U1q(0.551500655455414*pi,0.42242556439050016*pi) q[15];
U1q(0.726697548669963*pi,0.2362739139050305*pi) q[16];
U1q(0.308021157941392*pi,1.61664602221121*pi) q[17];
U1q(0.283494797903242*pi,1.2984227558871009*pi) q[18];
U1q(0.573938468092639*pi,1.4423004539677997*pi) q[19];
U1q(0.487989900912313*pi,0.3191114696517996*pi) q[20];
U1q(0.882233088139352*pi,1.9046018529733004*pi) q[21];
U1q(0.639083344041594*pi,1.9068545218888993*pi) q[22];
U1q(0.477274251334316*pi,1.8568268101822198*pi) q[23];
U1q(0.800562351152295*pi,1.5159406380652598*pi) q[24];
U1q(0.485516146089128*pi,0.048165337096589766*pi) q[25];
U1q(0.304107258395608*pi,0.18702805893508057*pi) q[26];
U1q(0.794319295478828*pi,1.179951233093*pi) q[27];
U1q(0.685569734489683*pi,1.4538818716227002*pi) q[28];
U1q(0.616682927311995*pi,0.42016760288141963*pi) q[29];
U1q(0.315400494168145*pi,0.9776041596388296*pi) q[30];
U1q(0.423446186408837*pi,0.59340382436333*pi) q[31];
U1q(0.257364103904509*pi,0.5929969132366093*pi) q[32];
U1q(0.781668066587356*pi,0.9676015815017003*pi) q[33];
U1q(0.873830609459667*pi,0.15001011778321072*pi) q[34];
U1q(0.487081772492998*pi,1.17163956570203*pi) q[35];
U1q(0.353504551904679*pi,1.0894447240788008*pi) q[36];
U1q(0.499340313282298*pi,1.6646375725633398*pi) q[37];
U1q(0.141044490288458*pi,0.11072257746029912*pi) q[38];
U1q(0.132495111884267*pi,0.4793914866362101*pi) q[39];
RZZ(0.5*pi) q[26],q[0];
RZZ(0.5*pi) q[1],q[21];
RZZ(0.5*pi) q[8],q[2];
RZZ(0.5*pi) q[3],q[9];
RZZ(0.5*pi) q[4],q[29];
RZZ(0.5*pi) q[5],q[28];
RZZ(0.5*pi) q[27],q[6];
RZZ(0.5*pi) q[7],q[30];
RZZ(0.5*pi) q[33],q[10];
RZZ(0.5*pi) q[11],q[25];
RZZ(0.5*pi) q[32],q[12];
RZZ(0.5*pi) q[31],q[13];
RZZ(0.5*pi) q[14],q[24];
RZZ(0.5*pi) q[17],q[15];
RZZ(0.5*pi) q[16],q[20];
RZZ(0.5*pi) q[39],q[18];
RZZ(0.5*pi) q[34],q[19];
RZZ(0.5*pi) q[22],q[35];
RZZ(0.5*pi) q[38],q[23];
RZZ(0.5*pi) q[36],q[37];
U1q(0.86144582055195*pi,0.7581668870347098*pi) q[0];
U1q(0.644134312580691*pi,0.24043608491050072*pi) q[1];
U1q(0.47504837863686*pi,1.4304671525490402*pi) q[2];
U1q(0.415067424017779*pi,0.9562057835467996*pi) q[3];
U1q(0.509578211916953*pi,1.0247400634515103*pi) q[4];
U1q(0.272451454021037*pi,0.9045601371911101*pi) q[5];
U1q(0.103149902261158*pi,0.3059681047021705*pi) q[6];
U1q(0.297182364017415*pi,0.5495955803899992*pi) q[7];
U1q(0.719460898655796*pi,1.2733744491077097*pi) q[8];
U1q(0.6499946740156*pi,1.8906813651639993*pi) q[9];
U1q(0.728494649342697*pi,1.9427885089210992*pi) q[10];
U1q(0.724874800156205*pi,1.8418546286705997*pi) q[11];
U1q(0.403363823853457*pi,0.28281613121952986*pi) q[12];
U1q(0.598382159111238*pi,1.1085600716618007*pi) q[13];
U1q(0.422993215310968*pi,1.7920077580074008*pi) q[14];
U1q(0.568166426375638*pi,0.6951287688745005*pi) q[15];
U1q(0.417938239659767*pi,1.3781230187322002*pi) q[16];
U1q(0.333015650087622*pi,0.23920666638725052*pi) q[17];
U1q(0.503586493923966*pi,1.8398204337746993*pi) q[18];
U1q(0.714790965582819*pi,0.8277442884163992*pi) q[19];
U1q(0.658470805468778*pi,1.4108732144621001*pi) q[20];
U1q(0.626939588104657*pi,0.6859069385990999*pi) q[21];
U1q(0.32334287023151*pi,0.8867787343178009*pi) q[22];
U1q(0.461430621710581*pi,1.0319197945269991*pi) q[23];
U1q(0.329336143965656*pi,1.6788894633934*pi) q[24];
U1q(0.846244222097182*pi,1.20607855144468*pi) q[25];
U1q(0.42264753275567*pi,0.6697539266387995*pi) q[26];
U1q(0.198244677650987*pi,0.6671591554701006*pi) q[27];
U1q(0.106351030184522*pi,1.4573925337899993*pi) q[28];
U1q(0.474587411227*pi,0.6786044823495594*pi) q[29];
U1q(0.441922547964818*pi,1.0755427957682002*pi) q[30];
U1q(0.509896931056249*pi,1.7426653284392*pi) q[31];
U1q(0.726246961472702*pi,0.9304899016491994*pi) q[32];
U1q(0.610248788116592*pi,0.8104629620146007*pi) q[33];
U1q(0.320470360722113*pi,1.3719771828635992*pi) q[34];
U1q(0.534612404255704*pi,1.7062329385205999*pi) q[35];
U1q(0.627977249518802*pi,0.7129202328238993*pi) q[36];
U1q(0.499284708675784*pi,1.03590794089207*pi) q[37];
U1q(0.444709114441012*pi,1.9511198989576997*pi) q[38];
U1q(0.458106666728054*pi,0.7677790356234997*pi) q[39];
RZZ(0.5*pi) q[4],q[0];
RZZ(0.5*pi) q[1],q[38];
RZZ(0.5*pi) q[15],q[2];
RZZ(0.5*pi) q[3],q[35];
RZZ(0.5*pi) q[5],q[9];
RZZ(0.5*pi) q[6],q[19];
RZZ(0.5*pi) q[7],q[23];
RZZ(0.5*pi) q[8],q[11];
RZZ(0.5*pi) q[21],q[10];
RZZ(0.5*pi) q[12],q[18];
RZZ(0.5*pi) q[32],q[13];
RZZ(0.5*pi) q[39],q[14];
RZZ(0.5*pi) q[16],q[31];
RZZ(0.5*pi) q[17],q[37];
RZZ(0.5*pi) q[20],q[36];
RZZ(0.5*pi) q[22],q[25];
RZZ(0.5*pi) q[34],q[24];
RZZ(0.5*pi) q[26],q[28];
RZZ(0.5*pi) q[33],q[27];
RZZ(0.5*pi) q[30],q[29];
U1q(0.556978201191514*pi,1.2021007433283*pi) q[0];
U1q(0.113776813997496*pi,1.0145245952527002*pi) q[1];
U1q(0.718750457674266*pi,0.42148176958630046*pi) q[2];
U1q(0.716702028605586*pi,1.7345115115750005*pi) q[3];
U1q(0.698763591147953*pi,1.3525721598123397*pi) q[4];
U1q(0.749638566230021*pi,1.4485913722554997*pi) q[5];
U1q(0.207183981062678*pi,1.5503243282038994*pi) q[6];
U1q(0.54908374141031*pi,0.24008143373930046*pi) q[7];
U1q(0.583973628934637*pi,0.21579140838778965*pi) q[8];
U1q(0.809309605416623*pi,0.9566968007192003*pi) q[9];
U1q(0.630368236034944*pi,0.7518300285218995*pi) q[10];
U1q(0.505095193446794*pi,1.1800247935365995*pi) q[11];
U1q(0.643555300835315*pi,1.7844769684741006*pi) q[12];
U1q(0.318161948336999*pi,0.8724934695664999*pi) q[13];
U1q(0.376731150930304*pi,1.1348878572544994*pi) q[14];
U1q(0.808994920183787*pi,1.3675386548150001*pi) q[15];
U1q(0.68161108550053*pi,1.5783100556453995*pi) q[16];
U1q(0.192858322117999*pi,1.9577544856386009*pi) q[17];
U1q(0.164273445942178*pi,1.2149935572792998*pi) q[18];
U1q(0.255063616282931*pi,1.1364346240094996*pi) q[19];
U1q(0.709860641088839*pi,1.0352995565373*pi) q[20];
U1q(0.420360062642852*pi,1.1841497659801004*pi) q[21];
U1q(0.52715258127134*pi,0.4964847598384008*pi) q[22];
U1q(0.630794392578477*pi,1.7492979870760994*pi) q[23];
U1q(0.666865974740692*pi,1.0775006511371998*pi) q[24];
U1q(0.286566225119989*pi,0.5983106501247999*pi) q[25];
U1q(0.606004365365203*pi,0.3766301321899004*pi) q[26];
U1q(0.254256151501429*pi,1.2842581257012*pi) q[27];
U1q(0.617952959433034*pi,0.6525642525483999*pi) q[28];
U1q(0.513120913123906*pi,1.5770910837350005*pi) q[29];
U1q(0.204153991860523*pi,1.4324763308088997*pi) q[30];
U1q(0.565996600182303*pi,0.5647717585144996*pi) q[31];
U1q(0.364876743569518*pi,1.7947740596338004*pi) q[32];
U1q(0.666629205084908*pi,0.8312738636524006*pi) q[33];
U1q(0.479289775693985*pi,0.9446467135137002*pi) q[34];
U1q(0.583403122386778*pi,0.7994844519310007*pi) q[35];
U1q(0.988566487959627*pi,1.5513852870651004*pi) q[36];
U1q(0.883526865972241*pi,1.6510813464877003*pi) q[37];
U1q(0.098829162522038*pi,0.631739447024799*pi) q[38];
U1q(0.447108327564789*pi,1.1701699940270007*pi) q[39];
RZZ(0.5*pi) q[31],q[0];
RZZ(0.5*pi) q[1],q[8];
RZZ(0.5*pi) q[20],q[2];
RZZ(0.5*pi) q[3],q[30];
RZZ(0.5*pi) q[26],q[4];
RZZ(0.5*pi) q[5],q[37];
RZZ(0.5*pi) q[6],q[29];
RZZ(0.5*pi) q[7],q[12];
RZZ(0.5*pi) q[11],q[9];
RZZ(0.5*pi) q[35],q[10];
RZZ(0.5*pi) q[21],q[13];
RZZ(0.5*pi) q[14],q[15];
RZZ(0.5*pi) q[16],q[39];
RZZ(0.5*pi) q[17],q[25];
RZZ(0.5*pi) q[18],q[38];
RZZ(0.5*pi) q[33],q[19];
RZZ(0.5*pi) q[22],q[24];
RZZ(0.5*pi) q[32],q[23];
RZZ(0.5*pi) q[27],q[34];
RZZ(0.5*pi) q[36],q[28];
U1q(0.251705209833784*pi,0.9162835807514007*pi) q[0];
U1q(0.570677575991119*pi,1.0230800006006007*pi) q[1];
U1q(0.704867318985322*pi,1.1314233685044996*pi) q[2];
U1q(0.191758776532193*pi,1.7189823319547983*pi) q[3];
U1q(0.444924606104738*pi,1.4080959155298007*pi) q[4];
U1q(0.513231724217937*pi,1.2475372893808991*pi) q[5];
U1q(0.582305829998568*pi,1.3543006077239*pi) q[6];
U1q(0.807628038618871*pi,1.3112262143924003*pi) q[7];
U1q(0.830127148444057*pi,1.163788825428*pi) q[8];
U1q(0.864409479600676*pi,1.3643794196377002*pi) q[9];
U1q(0.463790816576792*pi,0.9210502476555007*pi) q[10];
U1q(0.634077826070465*pi,0.4051518330274*pi) q[11];
U1q(0.811059162311741*pi,1.9832163591436007*pi) q[12];
U1q(0.497983258206318*pi,0.6461666103888994*pi) q[13];
U1q(0.470061194177433*pi,1.4834334024822997*pi) q[14];
U1q(0.493571039264416*pi,1.6738715070263002*pi) q[15];
U1q(0.350162901356301*pi,0.6933693448663014*pi) q[16];
U1q(0.409919471480077*pi,0.47054860818160016*pi) q[17];
U1q(0.680252289345695*pi,0.1329330799749009*pi) q[18];
U1q(0.570814336867147*pi,0.7960247407833982*pi) q[19];
U1q(0.714621896265712*pi,0.6844763171363013*pi) q[20];
U1q(0.587940001953667*pi,0.6678121612730017*pi) q[21];
U1q(0.253615399445431*pi,0.20892459198919866*pi) q[22];
U1q(0.972987244718245*pi,1.9535600378486997*pi) q[23];
U1q(0.346634788371888*pi,1.8130895225719001*pi) q[24];
U1q(0.735917024983132*pi,0.5703764653585992*pi) q[25];
U1q(0.806902479781862*pi,0.26381800631970087*pi) q[26];
U1q(0.090045228321568*pi,1.460548412303801*pi) q[27];
U1q(0.556918853690651*pi,0.8311026701349*pi) q[28];
U1q(0.403163617218482*pi,1.1142244650382995*pi) q[29];
U1q(0.226516470718885*pi,1.0354554360071013*pi) q[30];
U1q(0.360287335121302*pi,0.2708118455740003*pi) q[31];
U1q(0.67251583891923*pi,0.15558143962540072*pi) q[32];
U1q(0.288726995934586*pi,1.4822184081157985*pi) q[33];
U1q(0.742099678615941*pi,0.9336649966224009*pi) q[34];
U1q(0.266289994601272*pi,1.7324238564753003*pi) q[35];
U1q(0.750779314735026*pi,0.7926396630121992*pi) q[36];
U1q(0.482168706600598*pi,1.9476577317460002*pi) q[37];
U1q(0.689723084256714*pi,0.5290744111525996*pi) q[38];
U1q(0.550321153323286*pi,1.3683319487659986*pi) q[39];
RZZ(0.5*pi) q[17],q[0];
RZZ(0.5*pi) q[1],q[2];
RZZ(0.5*pi) q[3],q[22];
RZZ(0.5*pi) q[34],q[4];
RZZ(0.5*pi) q[14],q[5];
RZZ(0.5*pi) q[36],q[6];
RZZ(0.5*pi) q[7],q[9];
RZZ(0.5*pi) q[12],q[8];
RZZ(0.5*pi) q[11],q[10];
RZZ(0.5*pi) q[16],q[13];
RZZ(0.5*pi) q[15],q[29];
RZZ(0.5*pi) q[20],q[18];
RZZ(0.5*pi) q[28],q[19];
RZZ(0.5*pi) q[27],q[21];
RZZ(0.5*pi) q[37],q[23];
RZZ(0.5*pi) q[26],q[24];
RZZ(0.5*pi) q[31],q[25];
RZZ(0.5*pi) q[33],q[30];
RZZ(0.5*pi) q[39],q[32];
RZZ(0.5*pi) q[35],q[38];
U1q(0.556860932574488*pi,1.2463374440729993*pi) q[0];
U1q(0.93117049543337*pi,0.3304123726246999*pi) q[1];
U1q(0.864949943591718*pi,0.38718381446710026*pi) q[2];
U1q(0.594939419798175*pi,1.5030145399326997*pi) q[3];
U1q(0.687583060009321*pi,0.6316735956974*pi) q[4];
U1q(0.149870622326567*pi,1.6077535898003994*pi) q[5];
U1q(0.356806513605435*pi,1.2243843908492984*pi) q[6];
U1q(0.422805069037583*pi,0.24955515201379974*pi) q[7];
U1q(0.311850282193293*pi,0.11978548609610051*pi) q[8];
U1q(0.308191290893847*pi,0.4647912967268013*pi) q[9];
U1q(0.54251660948108*pi,0.4474194814663015*pi) q[10];
U1q(0.44410423884209*pi,0.7591940810109001*pi) q[11];
U1q(0.395504078513372*pi,1.8369807253249988*pi) q[12];
U1q(0.0655346618289616*pi,0.4922596902998997*pi) q[13];
U1q(0.69006312079247*pi,1.369226395574799*pi) q[14];
U1q(0.671005546271867*pi,0.7935758547699017*pi) q[15];
U1q(0.216438161237687*pi,0.10747135306920086*pi) q[16];
U1q(0.729806373138599*pi,1.3797888681311008*pi) q[17];
U1q(0.280840820538881*pi,1.6922845348255002*pi) q[18];
U1q(0.64441088421461*pi,1.2416995550460008*pi) q[19];
U1q(0.358783338603636*pi,0.5330498878871985*pi) q[20];
U1q(0.282561749219439*pi,1.7221953621440989*pi) q[21];
U1q(0.451997118086417*pi,0.9232342670089011*pi) q[22];
U1q(0.720717039367824*pi,1.9087896425204995*pi) q[23];
U1q(0.125492311938571*pi,1.3292121467899989*pi) q[24];
U1q(0.701872807644009*pi,1.2473862076467004*pi) q[25];
U1q(0.588415700722848*pi,1.6459288691352008*pi) q[26];
U1q(0.220304836071824*pi,0.26315548624830143*pi) q[27];
U1q(0.439609784615885*pi,0.4211440962997983*pi) q[28];
U1q(0.361692902474104*pi,1.4718344816645015*pi) q[29];
U1q(0.515567568778769*pi,0.1872933673786008*pi) q[30];
U1q(0.581750782597038*pi,1.5649238265513006*pi) q[31];
U1q(0.732022016589289*pi,0.18611925565859977*pi) q[32];
U1q(0.420285082610377*pi,1.661234953479699*pi) q[33];
U1q(0.880960333019201*pi,0.03731513957970023*pi) q[34];
U1q(0.572820026615856*pi,0.328402303875599*pi) q[35];
U1q(0.40561217910551*pi,0.48903113301219925*pi) q[36];
U1q(0.322065709375763*pi,1.3742522964246007*pi) q[37];
U1q(0.97427642853444*pi,1.497350488370401*pi) q[38];
U1q(0.675465579866448*pi,0.25782877010700034*pi) q[39];
rz(1.5264180425843001*pi) q[0];
rz(3.6714536231999*pi) q[1];
rz(1.9894748850060004*pi) q[2];
rz(0.12616257024849986*pi) q[3];
rz(0.9140656895286998*pi) q[4];
rz(2.142408639942399*pi) q[5];
rz(0.6708156771546001*pi) q[6];
rz(0.5859287362434991*pi) q[7];
rz(2.4555971434620005*pi) q[8];
rz(1.4044488314928998*pi) q[9];
rz(2.069669535058999*pi) q[10];
rz(1.7758114219991015*pi) q[11];
rz(1.2984504329718014*pi) q[12];
rz(1.4664738391647987*pi) q[13];
rz(0.9595026531500999*pi) q[14];
rz(1.9195609679800008*pi) q[15];
rz(0.496634550077399*pi) q[16];
rz(1.4302712591672986*pi) q[17];
rz(0.6990033930404991*pi) q[18];
rz(2.9429652792019994*pi) q[19];
rz(2.8195855918708013*pi) q[20];
rz(2.7988334211609*pi) q[21];
rz(3.8379650670825*pi) q[22];
rz(0.9551928863732009*pi) q[23];
rz(2.5319984585476014*pi) q[24];
rz(2.9919469687866*pi) q[25];
rz(3.352268944308001*pi) q[26];
rz(2.3672507409259005*pi) q[27];
rz(0.3215736288968003*pi) q[28];
rz(1.4795687402015005*pi) q[29];
rz(1.9487922450996003*pi) q[30];
rz(1.9680962949451999*pi) q[31];
rz(3.4282106574321993*pi) q[32];
rz(1.4709969207111016*pi) q[33];
rz(1.6770881812639011*pi) q[34];
rz(0.42716474533479953*pi) q[35];
rz(1.6484032500778003*pi) q[36];
rz(0.2593844174339992*pi) q[37];
rz(1.4083641926394996*pi) q[38];
rz(2.3109364995735007*pi) q[39];
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
