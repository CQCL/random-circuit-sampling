OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.544763822536048*pi,1.564874618659953*pi) q[0];
U1q(0.595006184402598*pi,0.422172843080432*pi) q[1];
U1q(0.456718360727498*pi,0.59550217652186*pi) q[2];
U1q(0.624855809807431*pi,1.60216446657076*pi) q[3];
U1q(0.678697646530725*pi,1.9658114549334496*pi) q[4];
U1q(0.712610221669133*pi,1.800148550874041*pi) q[5];
U1q(0.218646333987376*pi,0.407241968056657*pi) q[6];
U1q(0.510660888105338*pi,0.00689098056364967*pi) q[7];
U1q(0.882192510166759*pi,0.0746629766210888*pi) q[8];
U1q(0.051150183139103*pi,0.9643877578732101*pi) q[9];
U1q(0.185038780633392*pi,1.483422184376375*pi) q[10];
U1q(0.318768189948491*pi,1.091959171195262*pi) q[11];
U1q(0.541796550127601*pi,0.8489059723500401*pi) q[12];
U1q(0.446086414199106*pi,0.293267851102192*pi) q[13];
U1q(0.842828468091479*pi,1.1407719993494*pi) q[14];
U1q(0.831853041073239*pi,1.26642920969737*pi) q[15];
U1q(0.395355088370773*pi,0.101106394501513*pi) q[16];
U1q(0.470863760400829*pi,0.825000981502697*pi) q[17];
U1q(0.594928818689738*pi,0.194975809038097*pi) q[18];
U1q(0.531730924083931*pi,0.713474793345762*pi) q[19];
U1q(0.846444603676884*pi,0.654515579564241*pi) q[20];
U1q(0.477079287443829*pi,1.218363846005217*pi) q[21];
U1q(0.66114797010957*pi,1.44257942207241*pi) q[22];
U1q(0.574500321178603*pi,1.50023105035122*pi) q[23];
U1q(0.419927304584714*pi,0.988650904589054*pi) q[24];
U1q(0.470647930100427*pi,0.39476121398393005*pi) q[25];
U1q(0.686850757571281*pi,1.53189526288272*pi) q[26];
U1q(0.924440173215879*pi,0.615686309118655*pi) q[27];
U1q(0.238269674929494*pi,1.039604550570309*pi) q[28];
U1q(0.718701708570995*pi,1.159738747043056*pi) q[29];
U1q(0.309735187922853*pi,1.9729618923925523*pi) q[30];
U1q(0.909179634972344*pi,0.593331860207719*pi) q[31];
U1q(0.540312850362529*pi,0.663396365314127*pi) q[32];
U1q(0.721079712691713*pi,1.700559329273948*pi) q[33];
U1q(0.561800062243635*pi,0.85098159464012*pi) q[34];
U1q(0.169653436448714*pi,1.4765281296416561*pi) q[35];
U1q(0.718128290550601*pi,1.94686045764139*pi) q[36];
U1q(0.709227795242156*pi,0.745208038226189*pi) q[37];
U1q(0.386332680306492*pi,0.8949627819658601*pi) q[38];
U1q(0.379970746030777*pi,0.0559457446793258*pi) q[39];
RZZ(0.5*pi) q[0],q[8];
RZZ(0.5*pi) q[1],q[37];
RZZ(0.5*pi) q[2],q[19];
RZZ(0.5*pi) q[25],q[3];
RZZ(0.5*pi) q[4],q[24];
RZZ(0.5*pi) q[5],q[36];
RZZ(0.5*pi) q[6],q[13];
RZZ(0.5*pi) q[7],q[17];
RZZ(0.5*pi) q[9],q[14];
RZZ(0.5*pi) q[10],q[16];
RZZ(0.5*pi) q[34],q[11];
RZZ(0.5*pi) q[12],q[29];
RZZ(0.5*pi) q[28],q[15];
RZZ(0.5*pi) q[18],q[33];
RZZ(0.5*pi) q[20],q[27];
RZZ(0.5*pi) q[26],q[21];
RZZ(0.5*pi) q[39],q[22];
RZZ(0.5*pi) q[23],q[38];
RZZ(0.5*pi) q[30],q[32];
RZZ(0.5*pi) q[31],q[35];
U1q(0.802001652136969*pi,1.0400490006454501*pi) q[0];
U1q(0.261526477547889*pi,0.7316280473688801*pi) q[1];
U1q(0.399579824356115*pi,1.51490442893759*pi) q[2];
U1q(0.370647012144871*pi,0.70046928055217*pi) q[3];
U1q(0.848322832144792*pi,0.6343943965010999*pi) q[4];
U1q(0.655510216707398*pi,0.20336158293816986*pi) q[5];
U1q(0.413919514998164*pi,1.02092804798909*pi) q[6];
U1q(0.464522712786808*pi,1.4247312162196204*pi) q[7];
U1q(0.629679882809025*pi,1.4160357976332532*pi) q[8];
U1q(0.491889332914453*pi,0.1742844387692002*pi) q[9];
U1q(0.513913794919306*pi,0.2054466341422101*pi) q[10];
U1q(0.812475582497803*pi,0.6249746316188101*pi) q[11];
U1q(0.750953241566072*pi,0.82424180841561*pi) q[12];
U1q(0.406111687462859*pi,1.23008338665566*pi) q[13];
U1q(0.351041325979045*pi,1.6727561925597199*pi) q[14];
U1q(0.442186012762157*pi,0.620425556861547*pi) q[15];
U1q(0.404029211787988*pi,0.09947632998290001*pi) q[16];
U1q(0.3777199073067*pi,1.0114261046813802*pi) q[17];
U1q(0.661089617164657*pi,0.6146251265449001*pi) q[18];
U1q(0.447790561208805*pi,1.9508948395478698*pi) q[19];
U1q(0.161486608384929*pi,0.9604492827017199*pi) q[20];
U1q(0.675393628383915*pi,1.044872113386371*pi) q[21];
U1q(0.509728947944423*pi,0.650679791787025*pi) q[22];
U1q(0.345514932796609*pi,0.8193970799206101*pi) q[23];
U1q(0.772726412747131*pi,0.4465354959391199*pi) q[24];
U1q(0.845158665940962*pi,1.6643281022403502*pi) q[25];
U1q(0.713880213791877*pi,0.412785679138*pi) q[26];
U1q(0.401667160657124*pi,0.34707754382058*pi) q[27];
U1q(0.621635493232862*pi,0.6531503562668699*pi) q[28];
U1q(0.0673466535906263*pi,0.044769441048469805*pi) q[29];
U1q(0.337577809970194*pi,0.09515288832433*pi) q[30];
U1q(0.755761880040536*pi,1.97994571506957*pi) q[31];
U1q(0.37674546172898*pi,0.66704380169303*pi) q[32];
U1q(0.505330161693819*pi,1.5010135976938401*pi) q[33];
U1q(0.797063175611862*pi,1.9447411057420396*pi) q[34];
U1q(0.467647923065703*pi,1.8865172335043998*pi) q[35];
U1q(0.464082022685452*pi,0.7031777992571899*pi) q[36];
U1q(0.151014945539738*pi,0.72573889913699*pi) q[37];
U1q(0.82475911844646*pi,1.3803695441907*pi) q[38];
U1q(0.426548589142514*pi,1.3553834203111501*pi) q[39];
RZZ(0.5*pi) q[0],q[15];
RZZ(0.5*pi) q[1],q[38];
RZZ(0.5*pi) q[2],q[39];
RZZ(0.5*pi) q[9],q[3];
RZZ(0.5*pi) q[4],q[11];
RZZ(0.5*pi) q[5],q[16];
RZZ(0.5*pi) q[6],q[27];
RZZ(0.5*pi) q[7],q[35];
RZZ(0.5*pi) q[8],q[30];
RZZ(0.5*pi) q[10],q[26];
RZZ(0.5*pi) q[12],q[24];
RZZ(0.5*pi) q[13],q[37];
RZZ(0.5*pi) q[14],q[32];
RZZ(0.5*pi) q[17],q[29];
RZZ(0.5*pi) q[18],q[20];
RZZ(0.5*pi) q[21],q[19];
RZZ(0.5*pi) q[22],q[23];
RZZ(0.5*pi) q[36],q[25];
RZZ(0.5*pi) q[34],q[28];
RZZ(0.5*pi) q[31],q[33];
U1q(0.668097188595971*pi,0.8249507517033798*pi) q[0];
U1q(0.482886881391692*pi,1.7902728849245202*pi) q[1];
U1q(0.507050541803664*pi,1.2485720309272699*pi) q[2];
U1q(0.508457926290815*pi,1.1845365152615397*pi) q[3];
U1q(0.575120443073514*pi,1.7858283497100196*pi) q[4];
U1q(0.665206748438785*pi,1.9909550734149102*pi) q[5];
U1q(0.644489821057226*pi,0.7230894992500598*pi) q[6];
U1q(0.458730791667151*pi,1.5917248964736297*pi) q[7];
U1q(0.17426314261848*pi,1.6511959701201202*pi) q[8];
U1q(0.430322534340028*pi,0.2954167387565798*pi) q[9];
U1q(0.27664424450484*pi,1.10217527622454*pi) q[10];
U1q(0.381220213272338*pi,0.69464153937311*pi) q[11];
U1q(0.544715541349174*pi,0.4813272581597201*pi) q[12];
U1q(0.3858877307*pi,1.5434223654850001*pi) q[13];
U1q(0.73866982192893*pi,0.09327980443779005*pi) q[14];
U1q(0.718869661744622*pi,1.64049947689811*pi) q[15];
U1q(0.576435828509991*pi,1.77548079673259*pi) q[16];
U1q(0.709252941613627*pi,1.1115297900147603*pi) q[17];
U1q(0.737251196285206*pi,0.6978173589623502*pi) q[18];
U1q(0.230867193297993*pi,1.3126740635760799*pi) q[19];
U1q(0.108793283978753*pi,0.34785016129271007*pi) q[20];
U1q(0.46930131138981*pi,1.1468598501183198*pi) q[21];
U1q(0.462649852944875*pi,0.48314844551051994*pi) q[22];
U1q(0.353697566964738*pi,0.9473129906061901*pi) q[23];
U1q(0.772569386823567*pi,0.6815741497747498*pi) q[24];
U1q(0.172305389688372*pi,0.018790775117540015*pi) q[25];
U1q(0.558648317104677*pi,0.6957087057876301*pi) q[26];
U1q(0.331677566810654*pi,0.13813822919364993*pi) q[27];
U1q(0.700520901491991*pi,0.8271747263265001*pi) q[28];
U1q(0.724237643489692*pi,0.9676325796502203*pi) q[29];
U1q(0.57613297566404*pi,1.9818185941286997*pi) q[30];
U1q(0.646948934135411*pi,0.09320119445209984*pi) q[31];
U1q(0.438416280665994*pi,1.0133515391185304*pi) q[32];
U1q(0.802103076347248*pi,1.3456816076032299*pi) q[33];
U1q(0.535836052870314*pi,0.2288696367492804*pi) q[34];
U1q(0.831551542273468*pi,1.8706613374540497*pi) q[35];
U1q(0.647352893174001*pi,0.0494838792431902*pi) q[36];
U1q(0.448116744950987*pi,0.69361692098957*pi) q[37];
U1q(0.711424175380291*pi,0.14212216643919984*pi) q[38];
U1q(0.16558124764035*pi,0.40325826311732005*pi) q[39];
RZZ(0.5*pi) q[0],q[38];
RZZ(0.5*pi) q[1],q[25];
RZZ(0.5*pi) q[2],q[10];
RZZ(0.5*pi) q[12],q[3];
RZZ(0.5*pi) q[4],q[34];
RZZ(0.5*pi) q[5],q[22];
RZZ(0.5*pi) q[6],q[33];
RZZ(0.5*pi) q[7],q[24];
RZZ(0.5*pi) q[16],q[8];
RZZ(0.5*pi) q[9],q[27];
RZZ(0.5*pi) q[23],q[11];
RZZ(0.5*pi) q[13],q[30];
RZZ(0.5*pi) q[14],q[15];
RZZ(0.5*pi) q[17],q[35];
RZZ(0.5*pi) q[18],q[37];
RZZ(0.5*pi) q[29],q[19];
RZZ(0.5*pi) q[20],q[21];
RZZ(0.5*pi) q[26],q[36];
RZZ(0.5*pi) q[28],q[32];
RZZ(0.5*pi) q[31],q[39];
U1q(0.587691106412085*pi,0.11249671474356981*pi) q[0];
U1q(0.629527958727409*pi,0.2203547525525602*pi) q[1];
U1q(0.723144073890518*pi,0.7998048196191299*pi) q[2];
U1q(0.205169425256967*pi,1.9165667416519394*pi) q[3];
U1q(0.407256571461969*pi,1.3199463586300997*pi) q[4];
U1q(0.6553199269857*pi,0.11826199985906971*pi) q[5];
U1q(0.403858905663453*pi,0.9343597355262006*pi) q[6];
U1q(0.847299769167258*pi,0.6040883744315106*pi) q[7];
U1q(0.531046230182673*pi,0.7164032865869503*pi) q[8];
U1q(0.612128362219521*pi,1.1035673077071895*pi) q[9];
U1q(0.522602398509038*pi,1.35803248327601*pi) q[10];
U1q(0.319672570476908*pi,1.4951040071685302*pi) q[11];
U1q(0.446302414520301*pi,0.9167772098104603*pi) q[12];
U1q(0.387255323202098*pi,0.12211696655537985*pi) q[13];
U1q(0.243943106301794*pi,1.15397811408298*pi) q[14];
U1q(0.418900809323714*pi,1.8407613336251103*pi) q[15];
U1q(0.531889759129062*pi,1.7253904330829597*pi) q[16];
U1q(0.669878818165004*pi,0.17030406216097038*pi) q[17];
U1q(0.854486964281515*pi,1.96969984313625*pi) q[18];
U1q(0.501462092607774*pi,1.0899447998659797*pi) q[19];
U1q(0.220626997450854*pi,1.2999809337579*pi) q[20];
U1q(0.541254972555177*pi,0.17141426356294964*pi) q[21];
U1q(0.40952643592835*pi,1.11607178097477*pi) q[22];
U1q(0.326881415887985*pi,0.07502781073896969*pi) q[23];
U1q(0.0344107588832767*pi,1.2053520611821504*pi) q[24];
U1q(0.667372003297616*pi,0.12704058750851033*pi) q[25];
U1q(0.742841093895836*pi,1.9034427247806*pi) q[26];
U1q(0.247104576007169*pi,1.4903632432466498*pi) q[27];
U1q(0.59302194295897*pi,1.7467251819828098*pi) q[28];
U1q(0.397948738037223*pi,1.8221661580903508*pi) q[29];
U1q(0.549080075647024*pi,1.5361969014184602*pi) q[30];
U1q(0.363540833343309*pi,0.87424641602209*pi) q[31];
U1q(0.257488783187884*pi,1.43048363706502*pi) q[32];
U1q(0.194425445900277*pi,0.41473919169594*pi) q[33];
U1q(0.599765702615362*pi,0.32689543922159014*pi) q[34];
U1q(0.53845616139577*pi,0.48558397159604993*pi) q[35];
U1q(0.508161545640543*pi,0.0368284989358898*pi) q[36];
U1q(0.427147563384451*pi,1.7106882953576301*pi) q[37];
U1q(0.772221116010486*pi,1.5635229030512496*pi) q[38];
U1q(0.565804565756049*pi,0.82978148387655*pi) q[39];
RZZ(0.5*pi) q[0],q[26];
RZZ(0.5*pi) q[1],q[39];
RZZ(0.5*pi) q[2],q[23];
RZZ(0.5*pi) q[10],q[3];
RZZ(0.5*pi) q[4],q[21];
RZZ(0.5*pi) q[5],q[15];
RZZ(0.5*pi) q[6],q[18];
RZZ(0.5*pi) q[7],q[37];
RZZ(0.5*pi) q[34],q[8];
RZZ(0.5*pi) q[17],q[9];
RZZ(0.5*pi) q[11],q[28];
RZZ(0.5*pi) q[20],q[12];
RZZ(0.5*pi) q[13],q[16];
RZZ(0.5*pi) q[14],q[25];
RZZ(0.5*pi) q[19],q[30];
RZZ(0.5*pi) q[22],q[29];
RZZ(0.5*pi) q[24],q[32];
RZZ(0.5*pi) q[38],q[27];
RZZ(0.5*pi) q[31],q[36];
RZZ(0.5*pi) q[35],q[33];
U1q(0.54180078504369*pi,1.4533183724791696*pi) q[0];
U1q(0.521809120706625*pi,0.2631454027157698*pi) q[1];
U1q(0.70883996656405*pi,1.6028634162054*pi) q[2];
U1q(0.821094181463598*pi,1.8440023447057001*pi) q[3];
U1q(0.686825945199262*pi,0.7190516626430998*pi) q[4];
U1q(0.723897492015022*pi,0.5839163530340894*pi) q[5];
U1q(0.577078715780108*pi,0.44751517213565073*pi) q[6];
U1q(0.15535535817612*pi,1.9694479554087998*pi) q[7];
U1q(0.456270241087945*pi,0.3627111116453303*pi) q[8];
U1q(0.348980332147729*pi,1.2245958633920004*pi) q[9];
U1q(0.284168569714357*pi,1.6124894948555202*pi) q[10];
U1q(0.300674853546531*pi,0.39514003340754034*pi) q[11];
U1q(0.126833172944325*pi,1.1240102441925703*pi) q[12];
U1q(0.51560015153906*pi,0.9281229447039898*pi) q[13];
U1q(0.689369662677409*pi,0.4528014326736507*pi) q[14];
U1q(0.359442278907516*pi,1.18796895245527*pi) q[15];
U1q(0.256513636275952*pi,1.5993274749798996*pi) q[16];
U1q(0.709974565907495*pi,1.9604070964804006*pi) q[17];
U1q(0.668649920999331*pi,1.3540989497540696*pi) q[18];
U1q(0.456571927852795*pi,1.83913951999493*pi) q[19];
U1q(0.0758989477361172*pi,0.49822119193697034*pi) q[20];
U1q(0.150920157651255*pi,1.4097049005013496*pi) q[21];
U1q(0.247279270329558*pi,1.2190599019626598*pi) q[22];
U1q(0.131888159088733*pi,1.2558347033599002*pi) q[23];
U1q(0.664654358487818*pi,1.8394263810377804*pi) q[24];
U1q(0.0973537530707949*pi,0.9931098509046397*pi) q[25];
U1q(0.897079548943061*pi,1.67971461447023*pi) q[26];
U1q(0.970815772919853*pi,0.2060964066153801*pi) q[27];
U1q(0.595028706101962*pi,1.8820050516951596*pi) q[28];
U1q(0.316196469857235*pi,0.8668068711901995*pi) q[29];
U1q(0.112744174938409*pi,0.45402825181874995*pi) q[30];
U1q(0.720981444885223*pi,0.7004680411148101*pi) q[31];
U1q(0.240543010352998*pi,1.5601451825343098*pi) q[32];
U1q(0.678050779277336*pi,1.0597431562346706*pi) q[33];
U1q(0.0680269974321007*pi,0.81713736861016*pi) q[34];
U1q(0.977143459979427*pi,1.08476781352522*pi) q[35];
U1q(0.570655179414927*pi,1.5445277693621904*pi) q[36];
U1q(0.846895073992473*pi,1.7508270847802798*pi) q[37];
U1q(0.42434440660149*pi,1.5959310180073096*pi) q[38];
U1q(0.583943333770386*pi,1.5835365506671*pi) q[39];
RZZ(0.5*pi) q[0],q[24];
RZZ(0.5*pi) q[1],q[19];
RZZ(0.5*pi) q[2],q[3];
RZZ(0.5*pi) q[4],q[9];
RZZ(0.5*pi) q[5],q[25];
RZZ(0.5*pi) q[6],q[14];
RZZ(0.5*pi) q[7],q[13];
RZZ(0.5*pi) q[26],q[8];
RZZ(0.5*pi) q[10],q[32];
RZZ(0.5*pi) q[11],q[27];
RZZ(0.5*pi) q[12],q[36];
RZZ(0.5*pi) q[15],q[16];
RZZ(0.5*pi) q[17],q[38];
RZZ(0.5*pi) q[18],q[29];
RZZ(0.5*pi) q[20],q[33];
RZZ(0.5*pi) q[31],q[21];
RZZ(0.5*pi) q[35],q[22];
RZZ(0.5*pi) q[23],q[28];
RZZ(0.5*pi) q[37],q[30];
RZZ(0.5*pi) q[39],q[34];
U1q(0.613265652184597*pi,1.31931953781445*pi) q[0];
U1q(0.358762983012734*pi,0.1916486844025993*pi) q[1];
U1q(0.903444735234661*pi,0.18565348852879993*pi) q[2];
U1q(0.432166408654638*pi,1.3425209943679004*pi) q[3];
U1q(0.879920631281096*pi,0.3939335074664001*pi) q[4];
U1q(0.583531813650599*pi,1.4190400221417008*pi) q[5];
U1q(0.645673139112344*pi,0.7144915633188003*pi) q[6];
U1q(0.514355401015754*pi,1.6247855289579007*pi) q[7];
U1q(0.115787412835561*pi,1.8772504333600004*pi) q[8];
U1q(0.50322942902193*pi,1.3028659285700002*pi) q[9];
U1q(0.379337196261674*pi,1.6678118343037998*pi) q[10];
U1q(0.857158128426882*pi,0.08818967577489012*pi) q[11];
U1q(0.187754990532792*pi,0.5934596342221994*pi) q[12];
U1q(0.726857809124755*pi,1.8255765421481005*pi) q[13];
U1q(0.550633086721563*pi,0.3517629388719996*pi) q[14];
U1q(0.804540872916459*pi,1.1871374298231707*pi) q[15];
U1q(0.538309024648967*pi,1.3407695404309994*pi) q[16];
U1q(0.70195904233966*pi,1.9349047942728*pi) q[17];
U1q(0.772322533379915*pi,0.20074589860659042*pi) q[18];
U1q(0.652333718304982*pi,1.7870992721842*pi) q[19];
U1q(0.43403036772404*pi,1.3341976048247997*pi) q[20];
U1q(0.392480411562461*pi,0.8926877560602993*pi) q[21];
U1q(0.853296185208424*pi,0.30758349244051963*pi) q[22];
U1q(0.68050161894428*pi,0.7008649436866001*pi) q[23];
U1q(0.625473973396906*pi,0.37771193203164977*pi) q[24];
U1q(0.775392004726662*pi,0.6350481330131394*pi) q[25];
U1q(0.196338330865644*pi,1.8091168275946004*pi) q[26];
U1q(0.596291643197798*pi,0.9468488242495994*pi) q[27];
U1q(0.832716243472678*pi,1.5145024905183906*pi) q[28];
U1q(0.252570874841174*pi,1.6876636475191003*pi) q[29];
U1q(0.471960084525429*pi,0.21811515269689963*pi) q[30];
U1q(0.0572759784791264*pi,1.2499157143538007*pi) q[31];
U1q(0.332157816878366*pi,1.8257202483361006*pi) q[32];
U1q(0.263598262167975*pi,1.3672788676270997*pi) q[33];
U1q(0.336992145552031*pi,0.6639397946110002*pi) q[34];
U1q(0.56258348409179*pi,0.45692585260941065*pi) q[35];
U1q(0.324020978777357*pi,0.9297004041634995*pi) q[36];
U1q(0.778719079736512*pi,0.21407373025749976*pi) q[37];
U1q(0.383355069742557*pi,0.8522457516130801*pi) q[38];
U1q(0.607567796889194*pi,1.7826996845432994*pi) q[39];
RZZ(0.5*pi) q[0],q[25];
RZZ(0.5*pi) q[1],q[3];
RZZ(0.5*pi) q[2],q[11];
RZZ(0.5*pi) q[4],q[36];
RZZ(0.5*pi) q[18],q[5];
RZZ(0.5*pi) q[6],q[32];
RZZ(0.5*pi) q[7],q[15];
RZZ(0.5*pi) q[17],q[8];
RZZ(0.5*pi) q[9],q[16];
RZZ(0.5*pi) q[10],q[22];
RZZ(0.5*pi) q[26],q[12];
RZZ(0.5*pi) q[23],q[13];
RZZ(0.5*pi) q[28],q[14];
RZZ(0.5*pi) q[34],q[19];
RZZ(0.5*pi) q[20],q[37];
RZZ(0.5*pi) q[35],q[21];
RZZ(0.5*pi) q[38],q[24];
RZZ(0.5*pi) q[31],q[27];
RZZ(0.5*pi) q[29],q[30];
RZZ(0.5*pi) q[39],q[33];
U1q(0.566976747938683*pi,1.5190631556298406*pi) q[0];
U1q(0.850814390208843*pi,1.5558976845962995*pi) q[1];
U1q(0.461761984697629*pi,1.1913186732704997*pi) q[2];
U1q(0.481022316198329*pi,1.0201982734719*pi) q[3];
U1q(0.546530820837577*pi,0.9740923226876994*pi) q[4];
U1q(0.562139042621266*pi,0.5887638748802999*pi) q[5];
U1q(0.477436255977463*pi,0.5476413412801993*pi) q[6];
U1q(0.510054723824168*pi,1.4904696857879998*pi) q[7];
U1q(0.535301644066243*pi,0.2663862860019002*pi) q[8];
U1q(0.235243530831664*pi,1.6019878348811005*pi) q[9];
U1q(0.151038069219543*pi,1.1442758008034009*pi) q[10];
U1q(0.294330411812752*pi,0.8314517687012*pi) q[11];
U1q(0.115604556450583*pi,1.5698478774920002*pi) q[12];
U1q(0.366262892416987*pi,0.4569621502862997*pi) q[13];
U1q(0.480509262498281*pi,1.2334428219512006*pi) q[14];
U1q(0.665247269982582*pi,0.25337760886731076*pi) q[15];
U1q(0.28899541850368*pi,0.6021672854407996*pi) q[16];
U1q(0.682523617644705*pi,1.9414685867618005*pi) q[17];
U1q(0.214376657088771*pi,1.3092434350703996*pi) q[18];
U1q(0.750242010978636*pi,0.9609409122566994*pi) q[19];
U1q(0.303778763464849*pi,0.5360370693893994*pi) q[20];
U1q(0.267685550796761*pi,0.4789969105805003*pi) q[21];
U1q(0.734067365213601*pi,0.5509338932565804*pi) q[22];
U1q(0.281771145582125*pi,1.2078561566763995*pi) q[23];
U1q(0.338924875074124*pi,0.4090999014619996*pi) q[24];
U1q(0.363644177693509*pi,0.3121455410751004*pi) q[25];
U1q(0.63749771025995*pi,1.99936508706792*pi) q[26];
U1q(0.264403280279094*pi,0.41293784369610087*pi) q[27];
U1q(0.712915633519719*pi,1.1838339854592892*pi) q[28];
U1q(0.440202065335919*pi,0.9363754180794999*pi) q[29];
U1q(0.662474449504824*pi,0.33762339671189956*pi) q[30];
U1q(0.890719895263966*pi,0.7454803321511001*pi) q[31];
U1q(0.361788482865614*pi,1.6425896475158002*pi) q[32];
U1q(0.276769452926325*pi,1.9352950780401006*pi) q[33];
U1q(0.501173014949892*pi,0.3362249213202002*pi) q[34];
U1q(0.252857414951211*pi,0.21032727604320023*pi) q[35];
U1q(0.828786096497051*pi,0.9416163665656008*pi) q[36];
U1q(0.42244398057939*pi,1.4758570390856*pi) q[37];
U1q(0.422088826076858*pi,1.6030673854497994*pi) q[38];
U1q(0.131863121469793*pi,1.4037102058540007*pi) q[39];
RZZ(0.5*pi) q[5],q[0];
RZZ(0.5*pi) q[1],q[36];
RZZ(0.5*pi) q[2],q[15];
RZZ(0.5*pi) q[28],q[3];
RZZ(0.5*pi) q[4],q[29];
RZZ(0.5*pi) q[6],q[20];
RZZ(0.5*pi) q[7],q[21];
RZZ(0.5*pi) q[19],q[8];
RZZ(0.5*pi) q[9],q[30];
RZZ(0.5*pi) q[10],q[33];
RZZ(0.5*pi) q[11],q[38];
RZZ(0.5*pi) q[18],q[12];
RZZ(0.5*pi) q[22],q[13];
RZZ(0.5*pi) q[26],q[14];
RZZ(0.5*pi) q[37],q[16];
RZZ(0.5*pi) q[17],q[39];
RZZ(0.5*pi) q[23],q[34];
RZZ(0.5*pi) q[27],q[24];
RZZ(0.5*pi) q[35],q[25];
RZZ(0.5*pi) q[31],q[32];
U1q(0.770126219577534*pi,1.1938581119435199*pi) q[0];
U1q(0.773496746612733*pi,0.8791497608308987*pi) q[1];
U1q(0.804250084095418*pi,0.29300861176339943*pi) q[2];
U1q(0.769887775226813*pi,1.2376613826201002*pi) q[3];
U1q(0.578371038092894*pi,1.6099077965685993*pi) q[4];
U1q(0.745126163001954*pi,1.4361092175582009*pi) q[5];
U1q(0.330883710999468*pi,1.7686868847477015*pi) q[6];
U1q(0.020234098621664*pi,1.537034989563299*pi) q[7];
U1q(0.745874410304006*pi,1.8726927271411*pi) q[8];
U1q(0.305525377912281*pi,1.1490533199482016*pi) q[9];
U1q(0.611923026294678*pi,1.9056057589014*pi) q[10];
U1q(0.0909210691986198*pi,0.1008736022579999*pi) q[11];
U1q(0.236853802199043*pi,0.6958715682861012*pi) q[12];
U1q(0.295951851884895*pi,1.2678984402959994*pi) q[13];
U1q(0.899141525066307*pi,0.3804658725007002*pi) q[14];
U1q(0.406232638120451*pi,0.18727647978339945*pi) q[15];
U1q(0.569430212010339*pi,1.4101458162495*pi) q[16];
U1q(0.325507832632849*pi,0.39730307707510093*pi) q[17];
U1q(0.240418979041113*pi,1.0812020435239997*pi) q[18];
U1q(0.931810133914088*pi,1.2644264027888994*pi) q[19];
U1q(0.176514894305502*pi,1.794750222241401*pi) q[20];
U1q(0.770824791195636*pi,0.8311002274374992*pi) q[21];
U1q(0.393583489064087*pi,0.3549329745509109*pi) q[22];
U1q(0.149982180735283*pi,1.1641374322229012*pi) q[23];
U1q(0.845169466899487*pi,0.8951807555400002*pi) q[24];
U1q(0.296694034276531*pi,1.2568504204901991*pi) q[25];
U1q(0.910794722318877*pi,1.3813843493503501*pi) q[26];
U1q(0.403995339165356*pi,0.997566055188301*pi) q[27];
U1q(0.343341934251615*pi,0.14978309813320045*pi) q[28];
U1q(0.470705128746115*pi,0.7843013957409006*pi) q[29];
U1q(0.0673487887748459*pi,0.5045741149655001*pi) q[30];
U1q(0.855056838305377*pi,1.6140196263263995*pi) q[31];
U1q(0.746287245879518*pi,1.3316417256205995*pi) q[32];
U1q(0.369149200711511*pi,1.6505230324623987*pi) q[33];
U1q(0.593721755894228*pi,1.2799108098946004*pi) q[34];
U1q(0.0760586142579731*pi,1.6772092164355001*pi) q[35];
U1q(0.654868039360442*pi,1.8596220544635997*pi) q[36];
U1q(0.526309402555585*pi,0.19104074024810025*pi) q[37];
U1q(0.55483192680661*pi,0.3099256452518002*pi) q[38];
U1q(0.623825302352726*pi,1.5101484011547015*pi) q[39];
RZZ(0.5*pi) q[9],q[0];
RZZ(0.5*pi) q[1],q[31];
RZZ(0.5*pi) q[2],q[37];
RZZ(0.5*pi) q[16],q[3];
RZZ(0.5*pi) q[4],q[8];
RZZ(0.5*pi) q[5],q[10];
RZZ(0.5*pi) q[6],q[17];
RZZ(0.5*pi) q[7],q[28];
RZZ(0.5*pi) q[11],q[33];
RZZ(0.5*pi) q[12],q[32];
RZZ(0.5*pi) q[13],q[26];
RZZ(0.5*pi) q[14],q[29];
RZZ(0.5*pi) q[15],q[27];
RZZ(0.5*pi) q[18],q[35];
RZZ(0.5*pi) q[19],q[24];
RZZ(0.5*pi) q[23],q[20];
RZZ(0.5*pi) q[21],q[30];
RZZ(0.5*pi) q[22],q[34];
RZZ(0.5*pi) q[25],q[38];
RZZ(0.5*pi) q[39],q[36];
U1q(0.572568704614819*pi,1.1528290549490006*pi) q[0];
U1q(0.279127022005674*pi,0.47979763529869857*pi) q[1];
U1q(0.348239021310773*pi,1.236986387687299*pi) q[2];
U1q(0.340042725401324*pi,1.2662033755226005*pi) q[3];
U1q(0.780150973109506*pi,0.31787585138269847*pi) q[4];
U1q(0.558967818145298*pi,1.3926547131780005*pi) q[5];
U1q(0.105269327023474*pi,1.6429623768766*pi) q[6];
U1q(0.273974251587397*pi,0.7542874175406986*pi) q[7];
U1q(0.187566242040579*pi,1.186377454429799*pi) q[8];
U1q(0.352643388195434*pi,1.2507337825430014*pi) q[9];
U1q(0.163600609566565*pi,1.4468487171535003*pi) q[10];
U1q(0.494326644480134*pi,1.2696801771320985*pi) q[11];
U1q(0.380616430670986*pi,1.2001941013802018*pi) q[12];
U1q(0.790790432930425*pi,1.3685911488337013*pi) q[13];
U1q(0.140317269874037*pi,1.2052581522852002*pi) q[14];
U1q(0.203921855754119*pi,0.2347540141324007*pi) q[15];
U1q(0.811849673802648*pi,1.6645115166038984*pi) q[16];
U1q(0.566653509035624*pi,0.22373650630040132*pi) q[17];
U1q(0.34158379170553*pi,0.2019624811161993*pi) q[18];
U1q(0.622625360766401*pi,1.8156185493688*pi) q[19];
U1q(0.414950657455698*pi,0.8520720110213986*pi) q[20];
U1q(0.0376903571610836*pi,0.8259361624590014*pi) q[21];
U1q(0.613419161666147*pi,1.6015757119794998*pi) q[22];
U1q(0.33845565915808*pi,1.3249966114283005*pi) q[23];
U1q(0.51718288440246*pi,1.1781955301374012*pi) q[24];
U1q(0.470709593892731*pi,0.28544739185699974*pi) q[25];
U1q(0.622836739313288*pi,0.44733000512814947*pi) q[26];
U1q(0.442671621609986*pi,0.5598454163984989*pi) q[27];
U1q(0.193969896604669*pi,1.3771115880823999*pi) q[28];
U1q(0.532988794091621*pi,0.37167186028349875*pi) q[29];
U1q(0.19164239570057*pi,1.9601940879702013*pi) q[30];
U1q(0.140208728942978*pi,0.8835347059807006*pi) q[31];
U1q(0.446957631192946*pi,0.6615489253855991*pi) q[32];
U1q(0.813412251256157*pi,0.6502958024810006*pi) q[33];
U1q(0.407152530615987*pi,1.3247047850834015*pi) q[34];
U1q(0.733845747333579*pi,0.7129708632967002*pi) q[35];
U1q(0.0527534556559808*pi,0.9862828099325007*pi) q[36];
U1q(0.460014426131394*pi,0.6456439145539008*pi) q[37];
U1q(0.805245758215839*pi,0.06453431285899924*pi) q[38];
U1q(0.319463204799372*pi,0.47751506725780146*pi) q[39];
rz(2.929320814414*pi) q[0];
rz(3.5771842974981*pi) q[1];
rz(1.3511058405966985*pi) q[2];
rz(3.345171599307701*pi) q[3];
rz(1.4205335801065004*pi) q[4];
rz(1.721887555434499*pi) q[5];
rz(2.3150190409984006*pi) q[6];
rz(1.1091807443126989*pi) q[7];
rz(0.405135712261*pi) q[8];
rz(2.3300413219972995*pi) q[9];
rz(3.8560959466366*pi) q[10];
rz(2.2628950829441017*pi) q[11];
rz(0.8132161520635997*pi) q[12];
rz(0.04199533072710082*pi) q[13];
rz(3.1246091044110997*pi) q[14];
rz(2.5634889777831003*pi) q[15];
rz(2.9368915573503003*pi) q[16];
rz(2.0066564610223985*pi) q[17];
rz(2.1024831958962995*pi) q[18];
rz(1.5908991265303989*pi) q[19];
rz(3.5644526747027*pi) q[20];
rz(2.3424394258450008*pi) q[21];
rz(0.8145472777617009*pi) q[22];
rz(3.329161040956901*pi) q[23];
rz(1.5656708933037997*pi) q[24];
rz(1.228726625065999*pi) q[25];
rz(1.7629267576227399*pi) q[26];
rz(0.3249443900388016*pi) q[27];
rz(1.0101040405247002*pi) q[28];
rz(0.19090171622140062*pi) q[29];
rz(1.8785389993724984*pi) q[30];
rz(3.9355844621802003*pi) q[31];
rz(2.9072176531095018*pi) q[32];
rz(0.5040767480590986*pi) q[33];
rz(1.571131611729701*pi) q[34];
rz(2.1127906592498*pi) q[35];
rz(0.01626371124849868*pi) q[36];
rz(2.0009427052802007*pi) q[37];
rz(0.34271177019860133*pi) q[38];
rz(2.6673290933808005*pi) q[39];
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