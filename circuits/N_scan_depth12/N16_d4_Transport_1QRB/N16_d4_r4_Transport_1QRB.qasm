OPENQASM 2.0;
include "hqslib1.inc";

qreg q[16];
creg c[16];
rz(1.3915084764737*pi) q[0];
rz(3.779786139870377*pi) q[1];
rz(3.643113336132972*pi) q[2];
rz(2.3725765352582497*pi) q[3];
rz(1.1934155488879874*pi) q[4];
rz(3.68294556837267*pi) q[5];
rz(0.722888105869205*pi) q[6];
rz(1.12935181394488*pi) q[7];
rz(2.749462463613801*pi) q[8];
rz(1.31621050965326*pi) q[9];
rz(0.384029211768897*pi) q[10];
rz(1.8981399792417544*pi) q[11];
rz(3.9789981572694577*pi) q[12];
rz(0.31864198390686393*pi) q[13];
rz(1.24924259798857*pi) q[14];
rz(1.2361079561762378*pi) q[15];
U1q(0.604325973965315*pi,0.928364999593935*pi) q[0];
U1q(0.488980176870653*pi,0.321486268612942*pi) q[1];
U1q(0.161410732586539*pi,1.349564542165108*pi) q[2];
U1q(0.816785697980219*pi,1.297208118618653*pi) q[3];
U1q(1.27862160614935*pi,0.419881630650961*pi) q[4];
U1q(0.458500495027945*pi,1.4847271332623682*pi) q[5];
U1q(0.486027930608244*pi,0.017296565152687*pi) q[6];
U1q(0.117420251716401*pi,0.563822472805996*pi) q[7];
U1q(3.482684232117209*pi,1.653991401108962*pi) q[8];
U1q(0.706798387590453*pi,0.518987225215348*pi) q[9];
U1q(0.447916079250479*pi,1.434660565602877*pi) q[10];
U1q(1.26844329856982*pi,1.05911964698102*pi) q[11];
U1q(0.455173663276266*pi,0.039039762514251*pi) q[12];
U1q(3.337444330916572*pi,1.807080889639572*pi) q[13];
U1q(0.529502584390067*pi,0.419555327759243*pi) q[14];
U1q(1.50836683548476*pi,0.7846823484343*pi) q[15];
RZZ(0.0*pi) q[0],q[14];
RZZ(0.0*pi) q[10],q[1];
RZZ(0.0*pi) q[9],q[2];
RZZ(0.0*pi) q[3],q[5];
RZZ(0.0*pi) q[12],q[4];
RZZ(0.0*pi) q[11],q[6];
RZZ(0.0*pi) q[15],q[7];
RZZ(0.0*pi) q[13],q[8];
rz(2.55519619196849*pi) q[0];
rz(0.401682031456499*pi) q[1];
rz(0.420326372549379*pi) q[2];
rz(1.11490170552684*pi) q[3];
rz(0.0407522683067216*pi) q[4];
rz(0.912425710004224*pi) q[5];
rz(3.151625000868478*pi) q[6];
rz(3.870565705190239*pi) q[7];
rz(3.934998200047331*pi) q[8];
rz(3.789510868630228*pi) q[9];
rz(0.933583389637577*pi) q[10];
rz(0.224098297067695*pi) q[11];
rz(0.813373897710701*pi) q[12];
rz(1.27658382573012*pi) q[13];
rz(1.93384053370642*pi) q[14];
rz(0.0200182780084689*pi) q[15];
U1q(0.492548963664933*pi,1.893279076962718*pi) q[0];
U1q(0.507200290788906*pi,0.469565128366896*pi) q[1];
U1q(0.711376335180631*pi,0.887116246902443*pi) q[2];
U1q(0.446724010730535*pi,0.820808332651214*pi) q[3];
U1q(0.738959215513672*pi,0.0145047057907675*pi) q[4];
U1q(0.505433984618144*pi,0.825117996916262*pi) q[5];
U1q(0.764399457865665*pi,0.0437575261318978*pi) q[6];
U1q(0.788254300505716*pi,0.142062152793633*pi) q[7];
U1q(0.621325955142588*pi,0.389948342019895*pi) q[8];
U1q(0.342620542976298*pi,0.343201131010149*pi) q[9];
U1q(0.805094340032974*pi,0.515624541862154*pi) q[10];
U1q(0.413530870241665*pi,0.455205191002338*pi) q[11];
U1q(0.410520083456349*pi,0.781339322887019*pi) q[12];
U1q(0.311209231938212*pi,1.66614873477138*pi) q[13];
U1q(0.500728367016983*pi,1.46737633533035*pi) q[14];
U1q(0.62291113257522*pi,0.409982597527744*pi) q[15];
RZZ(0.0*pi) q[9],q[0];
RZZ(0.0*pi) q[2],q[1];
RZZ(0.0*pi) q[3],q[6];
RZZ(0.0*pi) q[4],q[11];
RZZ(0.0*pi) q[5],q[10];
RZZ(0.0*pi) q[14],q[7];
RZZ(0.0*pi) q[8],q[15];
RZZ(0.0*pi) q[12],q[13];
rz(1.07661171628809*pi) q[0];
rz(3.837981677385291*pi) q[1];
rz(0.77697220294358*pi) q[2];
rz(3.763718492239486*pi) q[3];
rz(3.595915168973434*pi) q[4];
rz(3.538863683271132*pi) q[5];
rz(0.674510428792368*pi) q[6];
rz(2.1791971169610598*pi) q[7];
rz(1.39512480173105*pi) q[8];
rz(2.74376983477112*pi) q[9];
rz(3.736769027489958*pi) q[10];
rz(2.6835895363491797*pi) q[11];
rz(2.0906391642690503*pi) q[12];
rz(3.825656210997148*pi) q[13];
rz(0.980066264611389*pi) q[14];
rz(2.24354997158951*pi) q[15];
U1q(0.296630358115537*pi,1.71966408176257*pi) q[0];
U1q(0.871764245739092*pi,0.478098861709976*pi) q[1];
U1q(0.483662737042633*pi,1.63397934618599*pi) q[2];
U1q(0.649509955729449*pi,0.0410561985306851*pi) q[3];
U1q(0.585781326175041*pi,0.334053563507194*pi) q[4];
U1q(0.258902685066873*pi,1.812242027549986*pi) q[5];
U1q(0.340550667636956*pi,1.12958356357246*pi) q[6];
U1q(0.867280086611215*pi,1.281298249362909*pi) q[7];
U1q(0.418560743653988*pi,1.16710642759367*pi) q[8];
U1q(0.567089183932931*pi,1.354128954813582*pi) q[9];
U1q(0.759052685676347*pi,0.141726082235954*pi) q[10];
U1q(0.578615435101639*pi,1.598087104647566*pi) q[11];
U1q(0.697089888706399*pi,1.474345392204236*pi) q[12];
U1q(0.304681874357981*pi,1.117782133320735*pi) q[13];
U1q(0.535000148972477*pi,0.337730491964705*pi) q[14];
U1q(0.911930527366049*pi,1.223624380241339*pi) q[15];
RZZ(0.0*pi) q[0],q[11];
RZZ(0.0*pi) q[14],q[1];
RZZ(0.0*pi) q[4],q[2];
RZZ(0.0*pi) q[13],q[3];
RZZ(0.0*pi) q[12],q[5];
RZZ(0.0*pi) q[6],q[7];
RZZ(0.0*pi) q[8],q[9];
RZZ(0.0*pi) q[10],q[15];
rz(0.192163041508813*pi) q[0];
rz(1.00240165391164*pi) q[1];
rz(3.745016357320321*pi) q[2];
rz(0.541523913692471*pi) q[3];
rz(3.796357343304252*pi) q[4];
rz(1.92250197141905*pi) q[5];
rz(0.0287808535713286*pi) q[6];
rz(0.664692913541667*pi) q[7];
rz(2.7598067725361*pi) q[8];
rz(0.485735268839944*pi) q[9];
rz(0.753993187084821*pi) q[10];
rz(3.25814725154466*pi) q[11];
rz(3.802629775660419*pi) q[12];
rz(0.223459970858034*pi) q[13];
rz(1.86364934057803*pi) q[14];
rz(0.0298873643144866*pi) q[15];
U1q(0.68632960006478*pi,0.382790076185402*pi) q[0];
U1q(0.435653227545997*pi,1.49214604423331*pi) q[1];
U1q(0.356288689662676*pi,0.590473546281795*pi) q[2];
U1q(0.35864964371362*pi,1.4908551977937021*pi) q[3];
U1q(0.299156595268179*pi,0.580601828104179*pi) q[4];
U1q(0.644363855219228*pi,1.43433724224881*pi) q[5];
U1q(0.467905183869622*pi,0.976141797856436*pi) q[6];
U1q(0.643356979898046*pi,0.616974429156986*pi) q[7];
U1q(0.678813841877663*pi,1.409386005986609*pi) q[8];
U1q(0.171731320613466*pi,1.531953364523194*pi) q[9];
U1q(0.434727476555731*pi,0.16844644970711*pi) q[10];
U1q(0.606461879802734*pi,1.4741427039503892*pi) q[11];
U1q(0.755666504791686*pi,1.8132291442887*pi) q[12];
U1q(0.618508435737521*pi,0.771983629941172*pi) q[13];
U1q(0.557746320531126*pi,1.28531091763992*pi) q[14];
U1q(0.189208345195324*pi,0.708439908811527*pi) q[15];
RZZ(0.0*pi) q[12],q[0];
RZZ(0.0*pi) q[4],q[1];
RZZ(0.0*pi) q[10],q[2];
RZZ(0.0*pi) q[3],q[14];
RZZ(0.0*pi) q[5],q[15];
RZZ(0.0*pi) q[9],q[6];
RZZ(0.0*pi) q[8],q[7];
RZZ(0.0*pi) q[13],q[11];
rz(2.01569612941795*pi) q[0];
rz(2.77074932496975*pi) q[1];
rz(0.865875480921239*pi) q[2];
rz(1.71576900662766*pi) q[3];
rz(1.19913085346299*pi) q[4];
rz(1.33022597129488*pi) q[5];
rz(2.08680009424711*pi) q[6];
rz(0.164989597978824*pi) q[7];
rz(0.983806412075908*pi) q[8];
rz(1.17463030711921*pi) q[9];
rz(2.13998770040935*pi) q[10];
rz(2.0951498248197398*pi) q[11];
rz(3.6527666840321222*pi) q[12];
rz(2.90042445933204*pi) q[13];
rz(1.76983795822434*pi) q[14];
rz(0.733677094563741*pi) q[15];
U1q(3.2154957299392493*pi,1.06146235839644*pi) q[0];
U1q(3.741953684179386*pi,0.5959441389886*pi) q[1];
U1q(3.381519050061451*pi,0.557509143272328*pi) q[2];
U1q(3.344875859844657*pi,1.797768084965991*pi) q[3];
U1q(3.70164335454391*pi,0.580569104609221*pi) q[4];
U1q(3.602714237693149*pi,1.56692130405921*pi) q[5];
U1q(3.232520264426188*pi,0.841514808961258*pi) q[6];
U1q(3.448926524580756*pi,0.842755359801265*pi) q[7];
U1q(3.708939778108256*pi,0.842521178670365*pi) q[8];
U1q(3.710295478632144*pi,0.0816589211432318*pi) q[9];
U1q(3.374162973374487*pi,0.421650828078741*pi) q[10];
U1q(3.793712177043331*pi,0.85163671328475*pi) q[11];
U1q(3.476664199781535*pi,1.771497146849828*pi) q[12];
U1q(3.592020108813429*pi,0.30325353820660994*pi) q[13];
U1q(3.9096407077193795*pi,1.00935376838296*pi) q[14];
U1q(3.468662497453418*pi,0.99979501151168*pi) q[15];
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