OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(3.528242786869513*pi,0.7191087410050403*pi) q[0];
U1q(1.95151323646559*pi,1.3615965979001974*pi) q[1];
U1q(1.28815576514243*pi,0.7877348756802461*pi) q[2];
U1q(0.418339915839295*pi,0.278168270341571*pi) q[3];
U1q(1.4352658874008*pi,0.32690784504153286*pi) q[4];
U1q(0.165962877156357*pi,1.9174196699701394*pi) q[5];
U1q(1.42184349344206*pi,1.4515317244634875*pi) q[6];
U1q(1.25435572650865*pi,1.718936933534031*pi) q[7];
U1q(1.44033325230709*pi,0.24599713323215988*pi) q[8];
U1q(0.355588778158576*pi,1.542251044306291*pi) q[9];
U1q(0.565975383149126*pi,0.31937152722796*pi) q[10];
U1q(0.472951997989892*pi,0.97172863492691*pi) q[11];
U1q(0.720437021063365*pi,1.768290058772011*pi) q[12];
U1q(0.29867925188798*pi,1.9517703296501907*pi) q[13];
U1q(1.57688407146312*pi,1.024968383925692*pi) q[14];
U1q(0.595298506907946*pi,0.407051673856782*pi) q[15];
U1q(0.411807753849817*pi,1.5164163151382621*pi) q[16];
U1q(0.387521472640214*pi,1.728788798197355*pi) q[17];
U1q(0.0939591103955607*pi,0.00181368806054127*pi) q[18];
U1q(0.461572364630195*pi,0.9397963552115101*pi) q[19];
U1q(0.651185699130498*pi,1.587237189761113*pi) q[20];
U1q(0.411953203193774*pi,0.31172192128211007*pi) q[21];
U1q(0.838110210123081*pi,0.308562452129314*pi) q[22];
U1q(0.70469761793426*pi,0.6577906834817*pi) q[23];
U1q(1.31937132065088*pi,1.5690823677609556*pi) q[24];
U1q(1.59517791334852*pi,1.7033064574198682*pi) q[25];
U1q(0.17949028033175*pi,0.236055206603236*pi) q[26];
U1q(1.61401979441796*pi,1.611648948835445*pi) q[27];
U1q(0.60373154296018*pi,0.747465652742132*pi) q[28];
U1q(0.480557863542659*pi,1.656196077762181*pi) q[29];
U1q(1.30706050084277*pi,0.613068574804735*pi) q[30];
U1q(0.473552015574584*pi,1.977086389264411*pi) q[31];
U1q(0.176351354491855*pi,0.8962469620831299*pi) q[32];
U1q(0.296980574577672*pi,1.698871170312196*pi) q[33];
U1q(0.511575436915757*pi,0.375997810691377*pi) q[34];
U1q(1.51752847664915*pi,1.704511616692908*pi) q[35];
U1q(1.7191053198122*pi,1.6630347180629053*pi) q[36];
U1q(0.310164565553696*pi,0.48739970937422*pi) q[37];
U1q(0.661590652883406*pi,0.653705561763997*pi) q[38];
U1q(0.457072729847022*pi,1.650197338429959*pi) q[39];
RZZ(0.5*pi) q[20],q[0];
RZZ(0.5*pi) q[28],q[1];
RZZ(0.5*pi) q[2],q[39];
RZZ(0.5*pi) q[11],q[3];
RZZ(0.5*pi) q[4],q[32];
RZZ(0.5*pi) q[5],q[24];
RZZ(0.5*pi) q[10],q[6];
RZZ(0.5*pi) q[23],q[7];
RZZ(0.5*pi) q[18],q[8];
RZZ(0.5*pi) q[9],q[14];
RZZ(0.5*pi) q[12],q[33];
RZZ(0.5*pi) q[13],q[27];
RZZ(0.5*pi) q[15],q[16];
RZZ(0.5*pi) q[26],q[17];
RZZ(0.5*pi) q[19],q[22];
RZZ(0.5*pi) q[34],q[21];
RZZ(0.5*pi) q[25],q[37];
RZZ(0.5*pi) q[29],q[31];
RZZ(0.5*pi) q[30],q[36];
RZZ(0.5*pi) q[35],q[38];
U1q(0.783661561522941*pi,1.9465208631802293*pi) q[0];
U1q(0.545798679333953*pi,1.9398348569155635*pi) q[1];
U1q(0.148775892911058*pi,1.0205915203980962*pi) q[2];
U1q(0.620535038604523*pi,0.07184669849102998*pi) q[3];
U1q(0.47020728784325*pi,1.737176271348043*pi) q[4];
U1q(0.361056044010018*pi,1.4202862691039102*pi) q[5];
U1q(0.359927010712439*pi,1.2658509682851276*pi) q[6];
U1q(0.77358877001758*pi,1.1631071811108207*pi) q[7];
U1q(0.502802042633838*pi,1.7320909998549898*pi) q[8];
U1q(0.390897315818344*pi,1.3049134610774402*pi) q[9];
U1q(0.431397824303609*pi,0.25210985695304*pi) q[10];
U1q(0.259264397160214*pi,0.9416000759898999*pi) q[11];
U1q(0.55386690997014*pi,1.37325759262803*pi) q[12];
U1q(0.255204112826944*pi,1.88737531716419*pi) q[13];
U1q(0.800223578607233*pi,1.2230972684259318*pi) q[14];
U1q(0.85013524624472*pi,1.088821175629817*pi) q[15];
U1q(0.469148767046911*pi,0.94363413485492*pi) q[16];
U1q(0.739638497824641*pi,1.39976146044281*pi) q[17];
U1q(0.465727434894902*pi,0.5663711177711699*pi) q[18];
U1q(0.161032453200154*pi,0.32230932943567003*pi) q[19];
U1q(0.448862535320856*pi,1.98010061013772*pi) q[20];
U1q(0.462134131671724*pi,1.24522034436835*pi) q[21];
U1q(0.61493188675014*pi,0.3505008071926099*pi) q[22];
U1q(0.716700580470459*pi,0.28060163910044*pi) q[23];
U1q(0.661620717429017*pi,0.4438600291650856*pi) q[24];
U1q(0.535793028248158*pi,1.2388649780668182*pi) q[25];
U1q(0.838679638269079*pi,1.25607885417093*pi) q[26];
U1q(0.376536075238109*pi,0.8365187023983149*pi) q[27];
U1q(0.655805621945422*pi,0.574610867779531*pi) q[28];
U1q(0.362354217906048*pi,1.6765611717523998*pi) q[29];
U1q(0.471253791339874*pi,0.8749237176108551*pi) q[30];
U1q(0.673637895695142*pi,1.3931785332636801*pi) q[31];
U1q(0.838969956227341*pi,0.0659934541938001*pi) q[32];
U1q(0.278046010574105*pi,1.5695141493090898*pi) q[33];
U1q(0.402605806008146*pi,1.201440930525178*pi) q[34];
U1q(0.426873877092678*pi,1.219976798595698*pi) q[35];
U1q(0.498860735081501*pi,1.001304060249895*pi) q[36];
U1q(0.339401956327089*pi,1.60024200457971*pi) q[37];
U1q(0.845373590986135*pi,1.066768274238739*pi) q[38];
U1q(0.65019257639347*pi,0.07628119523555998*pi) q[39];
RZZ(0.5*pi) q[9],q[0];
RZZ(0.5*pi) q[1],q[37];
RZZ(0.5*pi) q[2],q[11];
RZZ(0.5*pi) q[12],q[3];
RZZ(0.5*pi) q[4],q[22];
RZZ(0.5*pi) q[5],q[34];
RZZ(0.5*pi) q[33],q[6];
RZZ(0.5*pi) q[10],q[7];
RZZ(0.5*pi) q[8],q[24];
RZZ(0.5*pi) q[13],q[31];
RZZ(0.5*pi) q[14],q[38];
RZZ(0.5*pi) q[15],q[28];
RZZ(0.5*pi) q[32],q[16];
RZZ(0.5*pi) q[18],q[17];
RZZ(0.5*pi) q[19],q[35];
RZZ(0.5*pi) q[30],q[20];
RZZ(0.5*pi) q[29],q[21];
RZZ(0.5*pi) q[23],q[25];
RZZ(0.5*pi) q[26],q[27];
RZZ(0.5*pi) q[39],q[36];
U1q(0.778994946885144*pi,1.3724358602409104*pi) q[0];
U1q(0.285834532702029*pi,1.9926819131388176*pi) q[1];
U1q(0.597724717329709*pi,1.111291501843326*pi) q[2];
U1q(0.100164436372269*pi,0.8476524943814301*pi) q[3];
U1q(0.824244915792313*pi,1.3499287006366432*pi) q[4];
U1q(0.502960888399426*pi,0.6336620052671904*pi) q[5];
U1q(0.483374401605956*pi,0.5089600886025574*pi) q[6];
U1q(0.703469901102974*pi,1.3503578975910608*pi) q[7];
U1q(0.513975753575252*pi,0.9843720801746096*pi) q[8];
U1q(0.302808758626747*pi,1.7335164416847304*pi) q[9];
U1q(0.76583497577309*pi,1.4961279251543003*pi) q[10];
U1q(0.615424606029136*pi,1.3685856955006503*pi) q[11];
U1q(0.114727934642017*pi,1.91571536183731*pi) q[12];
U1q(0.767529931499685*pi,0.17457963426778011*pi) q[13];
U1q(0.410291732916435*pi,0.6085534492984213*pi) q[14];
U1q(0.574411128394579*pi,0.9402128714789799*pi) q[15];
U1q(0.497461261507061*pi,0.018224682715960228*pi) q[16];
U1q(0.68180186438864*pi,0.9372866089180598*pi) q[17];
U1q(0.616204683295025*pi,0.8768661869476801*pi) q[18];
U1q(0.577630074543209*pi,1.3027169813678698*pi) q[19];
U1q(0.584226623288436*pi,0.5023334593445901*pi) q[20];
U1q(0.809368494437724*pi,1.6289902963350196*pi) q[21];
U1q(0.42605383592255*pi,0.31985057774438985*pi) q[22];
U1q(0.156530682036207*pi,1.1387765889289403*pi) q[23];
U1q(0.423230809343907*pi,0.8665500648759958*pi) q[24];
U1q(0.240424271260291*pi,0.037600521159038*pi) q[25];
U1q(0.548880415015261*pi,1.8802832343276004*pi) q[26];
U1q(0.364013551395403*pi,1.1974342847178443*pi) q[27];
U1q(0.288510139563535*pi,0.62333848857962*pi) q[28];
U1q(0.779270479374323*pi,0.6529251042215098*pi) q[29];
U1q(0.736654918740394*pi,0.7490217202485749*pi) q[30];
U1q(0.498095835739704*pi,1.04691195511964*pi) q[31];
U1q(0.690808226826864*pi,1.2927903054117702*pi) q[32];
U1q(0.566097028203132*pi,1.9151297179858204*pi) q[33];
U1q(0.59332087942022*pi,1.3222161174982503*pi) q[34];
U1q(0.565399743998114*pi,0.21115395948225846*pi) q[35];
U1q(0.481802510598564*pi,1.5207789018275157*pi) q[36];
U1q(0.191977562123973*pi,0.5149040429155001*pi) q[37];
U1q(0.76504352580639*pi,0.60684393688474*pi) q[38];
U1q(0.55673711449936*pi,1.2204167514231097*pi) q[39];
RZZ(0.5*pi) q[0],q[31];
RZZ(0.5*pi) q[16],q[1];
RZZ(0.5*pi) q[2],q[36];
RZZ(0.5*pi) q[18],q[3];
RZZ(0.5*pi) q[4],q[35];
RZZ(0.5*pi) q[5],q[10];
RZZ(0.5*pi) q[37],q[6];
RZZ(0.5*pi) q[28],q[7];
RZZ(0.5*pi) q[26],q[8];
RZZ(0.5*pi) q[9],q[17];
RZZ(0.5*pi) q[39],q[11];
RZZ(0.5*pi) q[19],q[12];
RZZ(0.5*pi) q[13],q[38];
RZZ(0.5*pi) q[14],q[32];
RZZ(0.5*pi) q[15],q[30];
RZZ(0.5*pi) q[20],q[22];
RZZ(0.5*pi) q[21],q[27];
RZZ(0.5*pi) q[23],q[24];
RZZ(0.5*pi) q[25],q[33];
RZZ(0.5*pi) q[34],q[29];
U1q(0.312003332409721*pi,0.12773068180045*pi) q[0];
U1q(0.387737823233794*pi,0.8459664845793977*pi) q[1];
U1q(0.547223959813178*pi,1.0253718129277356*pi) q[2];
U1q(0.277661784845073*pi,1.8580172992876207*pi) q[3];
U1q(0.571874617123341*pi,0.23390349527202314*pi) q[4];
U1q(0.11370060882334*pi,0.9906663538362803*pi) q[5];
U1q(0.561189765066187*pi,0.3844782105210873*pi) q[6];
U1q(0.360119363486971*pi,0.8943194862087811*pi) q[7];
U1q(0.589370440943651*pi,1.8101458942113897*pi) q[8];
U1q(0.503725265879749*pi,0.18619405168634984*pi) q[9];
U1q(0.936082034851005*pi,1.9236267694273899*pi) q[10];
U1q(0.540708489086109*pi,1.51894309721958*pi) q[11];
U1q(0.618922960581288*pi,0.17688620107453978*pi) q[12];
U1q(0.747357994768809*pi,1.1592697375357703*pi) q[13];
U1q(0.530998855825277*pi,1.8874683128681724*pi) q[14];
U1q(0.212925174151892*pi,0.5001999093641096*pi) q[15];
U1q(0.708529265353186*pi,1.5838862617134595*pi) q[16];
U1q(0.361559547736277*pi,0.9582749010439704*pi) q[17];
U1q(0.418086133226499*pi,0.2829392423263606*pi) q[18];
U1q(0.334402714641203*pi,0.19737207816316982*pi) q[19];
U1q(0.192810481498145*pi,1.20087233189376*pi) q[20];
U1q(0.556580372180893*pi,0.9041928848448304*pi) q[21];
U1q(0.500027819768164*pi,1.8942086781865504*pi) q[22];
U1q(0.333799273349244*pi,0.96702876129208*pi) q[23];
U1q(0.12454728141848*pi,0.2305663743405857*pi) q[24];
U1q(0.641126230852763*pi,0.08174945934375799*pi) q[25];
U1q(0.0689296523201337*pi,1.7725258448358199*pi) q[26];
U1q(0.552029366803857*pi,0.7127531476204947*pi) q[27];
U1q(0.470156233068775*pi,1.1468742305271098*pi) q[28];
U1q(0.886544871337298*pi,1.3169743539695*pi) q[29];
U1q(0.342906971332794*pi,1.294426909914475*pi) q[30];
U1q(0.459250840227691*pi,1.0355125379816794*pi) q[31];
U1q(0.245363650419211*pi,0.9304409716105404*pi) q[32];
U1q(0.299516716759498*pi,1.4207074251622904*pi) q[33];
U1q(0.725442239650805*pi,1.3347075358851699*pi) q[34];
U1q(0.633498450812722*pi,1.328658729752929*pi) q[35];
U1q(0.838369575529938*pi,1.4688605429063957*pi) q[36];
U1q(0.134503987783652*pi,0.9235864516788697*pi) q[37];
U1q(0.56472119171259*pi,1.0826641986529202*pi) q[38];
U1q(0.532381821373291*pi,1.4611682920450697*pi) q[39];
RZZ(0.5*pi) q[11],q[0];
RZZ(0.5*pi) q[1],q[38];
RZZ(0.5*pi) q[2],q[16];
RZZ(0.5*pi) q[3],q[37];
RZZ(0.5*pi) q[26],q[4];
RZZ(0.5*pi) q[5],q[12];
RZZ(0.5*pi) q[15],q[6];
RZZ(0.5*pi) q[30],q[7];
RZZ(0.5*pi) q[8],q[9];
RZZ(0.5*pi) q[10],q[31];
RZZ(0.5*pi) q[17],q[13];
RZZ(0.5*pi) q[14],q[28];
RZZ(0.5*pi) q[18],q[23];
RZZ(0.5*pi) q[39],q[19];
RZZ(0.5*pi) q[20],q[24];
RZZ(0.5*pi) q[25],q[21];
RZZ(0.5*pi) q[22],q[32];
RZZ(0.5*pi) q[29],q[27];
RZZ(0.5*pi) q[33],q[35];
RZZ(0.5*pi) q[34],q[36];
U1q(0.336614702526409*pi,1.8010693754724105*pi) q[0];
U1q(0.0910679143334722*pi,0.24952320529949734*pi) q[1];
U1q(0.74486816375768*pi,0.07791016636706605*pi) q[2];
U1q(0.787155483220011*pi,1.6697102822082002*pi) q[3];
U1q(0.0394490962683396*pi,1.0810821849968129*pi) q[4];
U1q(0.166596761774005*pi,0.7969451258543803*pi) q[5];
U1q(0.57768930536591*pi,0.013752130623177372*pi) q[6];
U1q(0.38245466236699*pi,1.527416321639631*pi) q[7];
U1q(0.426007138662049*pi,1.5884333878542503*pi) q[8];
U1q(0.404089132567286*pi,1.5062832096055008*pi) q[9];
U1q(0.411204082639222*pi,1.9953981185841005*pi) q[10];
U1q(0.431914417997196*pi,0.3732531837865096*pi) q[11];
U1q(0.884898715356534*pi,0.2247210808206308*pi) q[12];
U1q(0.4113213977175*pi,0.05619619787510022*pi) q[13];
U1q(0.219249059006236*pi,0.40568373692075177*pi) q[14];
U1q(0.892006561574228*pi,0.5191983947769403*pi) q[15];
U1q(0.316135528218511*pi,0.6757676708869003*pi) q[16];
U1q(0.628733490289941*pi,0.37312877116201015*pi) q[17];
U1q(0.129856968911673*pi,1.5554807380250004*pi) q[18];
U1q(0.419476264681684*pi,1.4706422778913009*pi) q[19];
U1q(0.774690756786085*pi,1.2877999179280497*pi) q[20];
U1q(0.308759252326704*pi,0.10897349105315968*pi) q[21];
U1q(0.570534316551857*pi,0.30761551628158035*pi) q[22];
U1q(0.485363106839546*pi,1.3268831993028005*pi) q[23];
U1q(0.853608793349591*pi,1.2811251345367758*pi) q[24];
U1q(0.733095488809972*pi,1.8368302631672577*pi) q[25];
U1q(0.779101003899146*pi,0.22741639750100973*pi) q[26];
U1q(0.83739111311005*pi,0.928685901426725*pi) q[27];
U1q(0.416141208256343*pi,1.2430025550343604*pi) q[28];
U1q(0.758012557660494*pi,0.04117408472740003*pi) q[29];
U1q(0.111084902331932*pi,1.3249993743644062*pi) q[30];
U1q(0.668974022604938*pi,0.23742410266359926*pi) q[31];
U1q(0.642209265085431*pi,1.9133925281008697*pi) q[32];
U1q(0.474490399198149*pi,1.8027512648878297*pi) q[33];
U1q(0.178526828517055*pi,0.6558011510090402*pi) q[34];
U1q(0.686086204683902*pi,0.8108081074450091*pi) q[35];
U1q(0.884461333870775*pi,1.7005658964694348*pi) q[36];
U1q(0.640933194138525*pi,0.021487578809859187*pi) q[37];
U1q(0.463140999743367*pi,1.4001429234585103*pi) q[38];
U1q(0.411499790241945*pi,1.4653234249719809*pi) q[39];
RZZ(0.5*pi) q[0],q[35];
RZZ(0.5*pi) q[4],q[1];
RZZ(0.5*pi) q[2],q[6];
RZZ(0.5*pi) q[10],q[3];
RZZ(0.5*pi) q[5],q[25];
RZZ(0.5*pi) q[7],q[21];
RZZ(0.5*pi) q[8],q[19];
RZZ(0.5*pi) q[9],q[33];
RZZ(0.5*pi) q[11],q[38];
RZZ(0.5*pi) q[20],q[12];
RZZ(0.5*pi) q[13],q[29];
RZZ(0.5*pi) q[14],q[16];
RZZ(0.5*pi) q[15],q[37];
RZZ(0.5*pi) q[17],q[27];
RZZ(0.5*pi) q[18],q[26];
RZZ(0.5*pi) q[22],q[36];
RZZ(0.5*pi) q[23],q[34];
RZZ(0.5*pi) q[32],q[24];
RZZ(0.5*pi) q[39],q[28];
RZZ(0.5*pi) q[30],q[31];
U1q(0.473913484292515*pi,1.6853957568817304*pi) q[0];
U1q(0.57220313854313*pi,1.3461337556836277*pi) q[1];
U1q(0.202144211877439*pi,1.3008000450187467*pi) q[2];
U1q(0.266530357179622*pi,0.7443771504268*pi) q[3];
U1q(0.722034693928695*pi,1.4575113955802426*pi) q[4];
U1q(0.573417722994692*pi,1.4920792039667994*pi) q[5];
U1q(0.168213496057391*pi,0.7571828811999879*pi) q[6];
U1q(0.702793169804736*pi,1.1200084058533317*pi) q[7];
U1q(0.610869744639164*pi,1.5319521847785609*pi) q[8];
U1q(0.628836440122022*pi,0.8573671333877009*pi) q[9];
U1q(0.448272243532749*pi,1.6009491868508992*pi) q[10];
U1q(0.639662170774083*pi,1.4732135064125007*pi) q[11];
U1q(0.218593487675326*pi,0.5287626161546992*pi) q[12];
U1q(0.470233547554088*pi,0.9523722652493003*pi) q[13];
U1q(0.62797993254674*pi,0.14592238565329296*pi) q[14];
U1q(0.261830141412904*pi,1.8094098504626999*pi) q[15];
U1q(0.476814140579998*pi,1.5190618958641*pi) q[16];
U1q(0.223954644814007*pi,0.9530944071464598*pi) q[17];
U1q(0.0905052699079212*pi,1.2021901330723992*pi) q[18];
U1q(0.638269742070774*pi,0.3468272095396703*pi) q[19];
U1q(0.689442213499917*pi,1.4327189783706995*pi) q[20];
U1q(0.56057815981685*pi,1.1231491652245396*pi) q[21];
U1q(0.693557875009832*pi,1.1988674864379298*pi) q[22];
U1q(0.515250193963186*pi,0.46393430884319997*pi) q[23];
U1q(0.561430156745381*pi,0.1875029928731653*pi) q[24];
U1q(0.0976687622095674*pi,0.006858703391477761*pi) q[25];
U1q(0.197979884965245*pi,0.15116938828206017*pi) q[26];
U1q(0.689613927008032*pi,1.8165957836953446*pi) q[27];
U1q(0.593259025405671*pi,1.6178902399696895*pi) q[28];
U1q(0.249661824437806*pi,0.5820098131068301*pi) q[29];
U1q(0.613626624442202*pi,1.607523358239236*pi) q[30];
U1q(0.112017839246685*pi,0.8096787402839993*pi) q[31];
U1q(0.84110626179428*pi,1.5075179296787997*pi) q[32];
U1q(0.554768099909335*pi,1.6512909983441002*pi) q[33];
U1q(0.219766093984625*pi,1.7994741253047*pi) q[34];
U1q(0.149717869796884*pi,1.106282960802508*pi) q[35];
U1q(0.634049937738111*pi,1.8443986393446057*pi) q[36];
U1q(0.943624211036534*pi,0.3935129597951992*pi) q[37];
U1q(0.390304614833031*pi,0.6404060203075996*pi) q[38];
U1q(0.854196978179513*pi,1.4993107273340005*pi) q[39];
RZZ(0.5*pi) q[0],q[36];
RZZ(0.5*pi) q[12],q[1];
RZZ(0.5*pi) q[2],q[5];
RZZ(0.5*pi) q[8],q[3];
RZZ(0.5*pi) q[39],q[4];
RZZ(0.5*pi) q[35],q[6];
RZZ(0.5*pi) q[15],q[7];
RZZ(0.5*pi) q[9],q[22];
RZZ(0.5*pi) q[10],q[33];
RZZ(0.5*pi) q[11],q[14];
RZZ(0.5*pi) q[13],q[21];
RZZ(0.5*pi) q[18],q[16];
RZZ(0.5*pi) q[23],q[17];
RZZ(0.5*pi) q[25],q[19];
RZZ(0.5*pi) q[20],q[32];
RZZ(0.5*pi) q[29],q[24];
RZZ(0.5*pi) q[26],q[38];
RZZ(0.5*pi) q[27],q[37];
RZZ(0.5*pi) q[30],q[28];
RZZ(0.5*pi) q[34],q[31];
U1q(0.198631046689894*pi,1.13864614304876*pi) q[0];
U1q(0.345210910943634*pi,1.4618819231584066*pi) q[1];
U1q(0.489237844231434*pi,0.46122363889384665*pi) q[2];
U1q(0.341068050484512*pi,0.19439917265940032*pi) q[3];
U1q(0.504217780064737*pi,1.9328366172948321*pi) q[4];
U1q(0.609615868638185*pi,0.6966542555583004*pi) q[5];
U1q(0.503504771897125*pi,0.2679006004984874*pi) q[6];
U1q(0.342622869089854*pi,1.1846666760526325*pi) q[7];
U1q(0.207832615002017*pi,1.6043077618919597*pi) q[8];
U1q(0.431840998447183*pi,0.7977870475288*pi) q[9];
U1q(0.544742688621055*pi,0.5759928059902002*pi) q[10];
U1q(0.0818074009753304*pi,1.4814037293332003*pi) q[11];
U1q(0.327390372774406*pi,0.4528292872437003*pi) q[12];
U1q(0.3660750151275*pi,1.9714610917714008*pi) q[13];
U1q(0.30684328038156*pi,0.20589200731439306*pi) q[14];
U1q(0.759796786776271*pi,0.8395934729381995*pi) q[15];
U1q(0.415287333300495*pi,0.052018515226999185*pi) q[16];
U1q(0.21541585717901*pi,0.12312615555348927*pi) q[17];
U1q(0.626793141961837*pi,1.7668896661434985*pi) q[18];
U1q(0.184269882913538*pi,0.3197145594341997*pi) q[19];
U1q(0.475215449320048*pi,0.42394345491159946*pi) q[20];
U1q(0.285752631851679*pi,1.4337804481599008*pi) q[21];
U1q(0.366832502607074*pi,1.8638541130301007*pi) q[22];
U1q(0.212924316344311*pi,1.3976292482972*pi) q[23];
U1q(0.406508677469144*pi,0.9472505646729559*pi) q[24];
U1q(0.305829448503669*pi,1.1924377325085782*pi) q[25];
U1q(0.40488371404216*pi,0.4605408942842004*pi) q[26];
U1q(0.923451684994066*pi,1.1237495648237452*pi) q[27];
U1q(0.441771659034183*pi,1.1571301628260997*pi) q[28];
U1q(0.77556604010155*pi,0.08949361822569912*pi) q[29];
U1q(0.532342684165139*pi,1.6903626816446362*pi) q[30];
U1q(0.435627048641859*pi,0.2691199661318997*pi) q[31];
U1q(0.268625614838372*pi,0.22818604686050037*pi) q[32];
U1q(0.118748589027868*pi,0.49069154290880057*pi) q[33];
U1q(0.52830734226524*pi,1.9006466496848997*pi) q[34];
U1q(0.503815514072815*pi,1.1335634544521085*pi) q[35];
U1q(0.246403380417897*pi,0.12551886108230548*pi) q[36];
U1q(0.897973149434714*pi,0.4278307353913*pi) q[37];
U1q(0.883227664027029*pi,1.0495927674702994*pi) q[38];
U1q(0.514645424666668*pi,0.31077943525509966*pi) q[39];
rz(1.3221961685579586*pi) q[0];
rz(3.8256383572688026*pi) q[1];
rz(1.4128464575603523*pi) q[2];
rz(0.9248432329648004*pi) q[3];
rz(0.6656212519593687*pi) q[4];
rz(3.8453339319585993*pi) q[5];
rz(3.5307856003201117*pi) q[6];
rz(3.4428346720039684*pi) q[7];
rz(2.07801629124344*pi) q[8];
rz(1.2778924521453003*pi) q[9];
rz(3.593940358788*pi) q[10];
rz(2.1363863022230998*pi) q[11];
rz(2.6930025404494007*pi) q[12];
rz(1.7390614886026992*pi) q[13];
rz(1.8878256462146084*pi) q[14];
rz(2.875955316155199*pi) q[15];
rz(0.32771785412930043*pi) q[16];
rz(2.720317879802*pi) q[17];
rz(1.7013632698523011*pi) q[18];
rz(3.1315585597990996*pi) q[19];
rz(0.36044757090079926*pi) q[20];
rz(2.6497105808812*pi) q[21];
rz(3.1860303791040003*pi) q[22];
rz(3.0865954020485997*pi) q[23];
rz(0.46925936938984414*pi) q[24];
rz(1.7837339946456208*pi) q[25];
rz(0.1397350335882006*pi) q[26];
rz(2.493394764297456*pi) q[27];
rz(2.337354202295*pi) q[28];
rz(2.6663620833774004*pi) q[29];
rz(2.166585916851565*pi) q[30];
rz(2.2546354425794988*pi) q[31];
rz(1.4581033015290998*pi) q[32];
rz(0.49298648523739885*pi) q[33];
rz(0.4608395973002999*pi) q[34];
rz(1.7265679256194915*pi) q[35];
rz(0.2565336227680959*pi) q[36];
rz(1.662246183058901*pi) q[37];
rz(0.9229241464596996*pi) q[38];
rz(1.2178292778273008*pi) q[39];
U1q(0.198631046689894*pi,1.460842311606757*pi) q[0];
U1q(3.345210910943634*pi,0.287520280427196*pi) q[1];
U1q(0.489237844231434*pi,0.8740700964541499*pi) q[2];
U1q(1.34106805048451*pi,0.119242405624271*pi) q[3];
U1q(0.504217780064737*pi,1.598457869254208*pi) q[4];
U1q(1.60961586863819*pi,1.541988187516869*pi) q[5];
U1q(0.503504771897125*pi,0.798686200818583*pi) q[6];
U1q(0.342622869089854*pi,1.627501348056604*pi) q[7];
U1q(0.207832615002017*pi,0.682324053135369*pi) q[8];
U1q(1.43184099844718*pi,1.07567949967419*pi) q[9];
U1q(0.544742688621055*pi,1.169933164778164*pi) q[10];
U1q(1.08180740097533*pi,0.617790031556346*pi) q[11];
U1q(1.32739037277441*pi,0.145831827693065*pi) q[12];
U1q(1.3660750151275*pi,0.71052258037414*pi) q[13];
U1q(1.30684328038156*pi,1.09371765352901*pi) q[14];
U1q(1.75979678677627*pi,0.715548789093474*pi) q[15];
U1q(0.415287333300495*pi,1.379736369356352*pi) q[16];
U1q(0.21541585717901*pi,1.843444035355515*pi) q[17];
U1q(3.626793141961837*pi,0.468252935995833*pi) q[18];
U1q(1.18426988291354*pi,0.451273119233288*pi) q[19];
U1q(1.47521544932005*pi,1.784391025812388*pi) q[20];
U1q(1.28575263185168*pi,1.0834910290411*pi) q[21];
U1q(0.366832502607074*pi,0.0498844921341655*pi) q[22];
U1q(1.21292431634431*pi,1.484224650345813*pi) q[23];
U1q(1.40650867746914*pi,0.416509934062803*pi) q[24];
U1q(1.30582944850367*pi,1.976171727154197*pi) q[25];
U1q(1.40488371404216*pi,1.600275927872405*pi) q[26];
U1q(1.92345168499407*pi,0.617144329121265*pi) q[27];
U1q(0.441771659034183*pi,0.494484365121113*pi) q[28];
U1q(1.77556604010155*pi,1.755855701603076*pi) q[29];
U1q(0.532342684165139*pi,0.85694859849613*pi) q[30];
U1q(1.43562704864186*pi,1.523755408711369*pi) q[31];
U1q(1.26862561483837*pi,0.686289348389594*pi) q[32];
U1q(0.118748589027868*pi,1.9836780281461561*pi) q[33];
U1q(0.52830734226524*pi,1.361486246985202*pi) q[34];
U1q(0.503815514072815*pi,1.860131380071524*pi) q[35];
U1q(1.2464033804179*pi,1.382052483850479*pi) q[36];
U1q(1.89797314943471*pi,1.090076918450189*pi) q[37];
U1q(1.88322766402703*pi,0.972516913929972*pi) q[38];
U1q(1.51464542466667*pi,0.528608713082333*pi) q[39];
RZZ(0.5*pi) q[0],q[36];
RZZ(0.5*pi) q[12],q[1];
RZZ(0.5*pi) q[2],q[5];
RZZ(0.5*pi) q[8],q[3];
RZZ(0.5*pi) q[39],q[4];
RZZ(0.5*pi) q[35],q[6];
RZZ(0.5*pi) q[15],q[7];
RZZ(0.5*pi) q[9],q[22];
RZZ(0.5*pi) q[10],q[33];
RZZ(0.5*pi) q[11],q[14];
RZZ(0.5*pi) q[13],q[21];
RZZ(0.5*pi) q[18],q[16];
RZZ(0.5*pi) q[23],q[17];
RZZ(0.5*pi) q[25],q[19];
RZZ(0.5*pi) q[20],q[32];
RZZ(0.5*pi) q[29],q[24];
RZZ(0.5*pi) q[26],q[38];
RZZ(0.5*pi) q[27],q[37];
RZZ(0.5*pi) q[30],q[28];
RZZ(0.5*pi) q[34],q[31];
U1q(0.473913484292515*pi,0.00759192543972897*pi) q[0];
U1q(3.42779686145687*pi,0.4032684479019786*pi) q[1];
U1q(1.20214421187744*pi,1.71364650257902*pi) q[2];
U1q(3.733469642820377*pi,0.5692644278569267*pi) q[3];
U1q(1.7220346939287*pi,1.1231326475396002*pi) q[4];
U1q(1.57341772299469*pi,0.7465632391083523*pi) q[5];
U1q(0.168213496057391*pi,0.287968481520034*pi) q[6];
U1q(1.70279316980474*pi,1.5628430778573499*pi) q[7];
U1q(0.610869744639164*pi,0.609968476021975*pi) q[8];
U1q(3.3711635598779788*pi,0.016099413815317792*pi) q[9];
U1q(1.44827224353275*pi,1.19488954563887*pi) q[10];
U1q(3.3603378292259167*pi,0.6259802544771143*pi) q[11];
U1q(1.21859348767533*pi,1.0698984987820919*pi) q[12];
U1q(3.470233547554089*pi,0.7296114068962414*pi) q[13];
U1q(1.62797993254674*pi,0.1536872751901373*pi) q[14];
U1q(3.738169858587095*pi,0.7457324115690278*pi) q[15];
U1q(0.476814140579998*pi,0.846779749993455*pi) q[16];
U1q(0.223954644814007*pi,0.67341228694849*pi) q[17];
U1q(1.09050526990792*pi,1.0329524690669125*pi) q[18];
U1q(3.361730257929226*pi,0.4241604691277884*pi) q[19];
U1q(3.689442213499917*pi,1.7756155023533333*pi) q[20];
U1q(1.56057815981685*pi,0.39412231197643977*pi) q[21];
U1q(1.69355787500983*pi,1.384897865541956*pi) q[22];
U1q(3.484749806036814*pi,1.4179195897998809*pi) q[23];
U1q(3.4385698432546192*pi,1.1762575058626146*pi) q[24];
U1q(3.902331237790431*pi,1.161750756271294*pi) q[25];
U1q(1.19797988496524*pi,0.9096474338745562*pi) q[26];
U1q(1.68961392700803*pi,0.9242981102497236*pi) q[27];
U1q(1.59325902540567*pi,1.9552444422646902*pi) q[28];
U1q(1.24966182443781*pi,0.2633395067219426*pi) q[29];
U1q(3.613626624442203*pi,1.7741092750907201*pi) q[30];
U1q(3.112017839246684*pi,0.9831966345593204*pi) q[31];
U1q(1.84110626179428*pi,0.4069574655713205*pi) q[32];
U1q(0.554768099909335*pi,0.14427748358143*pi) q[33];
U1q(1.21976609398462*pi,1.26031372260509*pi) q[34];
U1q(0.149717869796884*pi,0.8328508864219999*pi) q[35];
U1q(3.3659500622618888*pi,0.6631727055882035*pi) q[36];
U1q(1.94362421103653*pi,0.12439469404628323*pi) q[37];
U1q(3.609695385166969*pi,1.38170366109261*pi) q[38];
U1q(1.85419697817951*pi,1.340077421003411*pi) q[39];
RZZ(0.5*pi) q[0],q[35];
RZZ(0.5*pi) q[4],q[1];
RZZ(0.5*pi) q[2],q[6];
RZZ(0.5*pi) q[10],q[3];
RZZ(0.5*pi) q[5],q[25];
RZZ(0.5*pi) q[7],q[21];
RZZ(0.5*pi) q[8],q[19];
RZZ(0.5*pi) q[9],q[33];
RZZ(0.5*pi) q[11],q[38];
RZZ(0.5*pi) q[20],q[12];
RZZ(0.5*pi) q[13],q[29];
RZZ(0.5*pi) q[14],q[16];
RZZ(0.5*pi) q[15],q[37];
RZZ(0.5*pi) q[17],q[27];
RZZ(0.5*pi) q[18],q[26];
RZZ(0.5*pi) q[22],q[36];
RZZ(0.5*pi) q[23],q[34];
RZZ(0.5*pi) q[32],q[24];
RZZ(0.5*pi) q[39],q[28];
RZZ(0.5*pi) q[30],q[31];
U1q(1.33661470252641*pi,1.123265544030406*pi) q[0];
U1q(3.9089320856665277*pi,0.4998789982861045*pi) q[1];
U1q(3.2551318362423203*pi,1.936536381230653*pi) q[2];
U1q(1.78715548322001*pi,1.6439312960755443*pi) q[3];
U1q(3.9605509037316615*pi,1.4995618581230254*pi) q[4];
U1q(0.166596761774005*pi,0.05142916099590322*pi) q[5];
U1q(0.57768930536591*pi,1.5445377309432602*pi) q[6];
U1q(1.38245466236699*pi,1.1554351620710688*pi) q[7];
U1q(1.42600713866205*pi,0.6664496790976799*pi) q[8];
U1q(3.595910867432714*pi,1.3671833375975249*pi) q[9];
U1q(1.41120408263922*pi,1.8004406139056934*pi) q[10];
U1q(3.568085582002804*pi,0.7259405771030866*pi) q[11];
U1q(1.88489871535653*pi,1.7658569634480368*pi) q[12];
U1q(1.4113213977175*pi,0.8334353395220511*pi) q[13];
U1q(1.21924905900624*pi,1.4134486264576251*pi) q[14];
U1q(3.892006561574229*pi,1.0359438672547796*pi) q[15];
U1q(1.31613552821851*pi,1.003485525016289*pi) q[16];
U1q(1.62873349028994*pi,1.09344665096403*pi) q[17];
U1q(3.129856968911673*pi,0.38624307401946556*pi) q[18];
U1q(3.580523735318316*pi,1.3003454007761652*pi) q[19];
U1q(1.77469075678609*pi,0.6306964419107235*pi) q[20];
U1q(0.308759252326704*pi,1.379946637805058*pi) q[21];
U1q(3.429465683448143*pi,0.27614983569831564*pi) q[22];
U1q(3.514636893160454*pi,1.5549706993402226*pi) q[23];
U1q(3.853608793349592*pi,1.0826353641990014*pi) q[24];
U1q(1.73309548880997*pi,0.3317791964955228*pi) q[25];
U1q(1.77910100389915*pi,1.9858944430934962*pi) q[26];
U1q(0.83739111311005*pi,1.036388227981139*pi) q[27];
U1q(3.583858791743656*pi,1.3301321272000308*pi) q[28];
U1q(0.758012557660494*pi,0.722503778342503*pi) q[29];
U1q(3.888915097668067*pi,1.056633258965534*pi) q[30];
U1q(3.6689740226049388*pi,0.4109419969389614*pi) q[31];
U1q(1.64220926508543*pi,0.8128320639933766*pi) q[32];
U1q(3.474490399198149*pi,0.2957377501252001*pi) q[33];
U1q(3.821473171482944*pi,0.40398669690079547*pi) q[34];
U1q(3.6860862046839022*pi,1.53737603306441*pi) q[35];
U1q(3.115538666129225*pi,1.8070054484633857*pi) q[36];
U1q(0.640933194138525*pi,1.7523693130609233*pi) q[37];
U1q(3.536859000256633*pi,1.6219667579417196*pi) q[38];
U1q(0.411499790241945*pi,1.306090118641408*pi) q[39];
RZZ(0.5*pi) q[11],q[0];
RZZ(0.5*pi) q[1],q[38];
RZZ(0.5*pi) q[2],q[16];
RZZ(0.5*pi) q[3],q[37];
RZZ(0.5*pi) q[26],q[4];
RZZ(0.5*pi) q[5],q[12];
RZZ(0.5*pi) q[15],q[6];
RZZ(0.5*pi) q[30],q[7];
RZZ(0.5*pi) q[8],q[9];
RZZ(0.5*pi) q[10],q[31];
RZZ(0.5*pi) q[17],q[13];
RZZ(0.5*pi) q[14],q[28];
RZZ(0.5*pi) q[18],q[23];
RZZ(0.5*pi) q[39],q[19];
RZZ(0.5*pi) q[20],q[24];
RZZ(0.5*pi) q[25],q[21];
RZZ(0.5*pi) q[22],q[32];
RZZ(0.5*pi) q[29],q[27];
RZZ(0.5*pi) q[33],q[35];
RZZ(0.5*pi) q[34],q[36];
U1q(3.312003332409721*pi,1.7966042377023683*pi) q[0];
U1q(1.38773782323379*pi,1.903435719006211*pi) q[1];
U1q(3.452776040186822*pi,1.989074734669983*pi) q[2];
U1q(1.27766178484507*pi,0.8322383131549942*pi) q[3];
U1q(3.428125382876659*pi,1.3467405478478156*pi) q[4];
U1q(0.11370060882334*pi,0.24515038897780306*pi) q[5];
U1q(1.56118976506619*pi,0.9152638108411701*pi) q[6];
U1q(1.36011936348697*pi,0.5223383266402091*pi) q[7];
U1q(3.589370440943651*pi,1.4447371727405436*pi) q[8];
U1q(3.496274734120251*pi,1.6872724955166798*pi) q[9];
U1q(0.936082034851005*pi,1.7286692647489836*pi) q[10];
U1q(3.54070848908611*pi,1.580250663670007*pi) q[11];
U1q(3.618922960581288*pi,1.8136918431941265*pi) q[12];
U1q(3.25264200523119*pi,1.7303617998613738*pi) q[13];
U1q(1.53099885582528*pi,0.9316640505102103*pi) q[14];
U1q(1.21292517415189*pi,0.016945381841949825*pi) q[15];
U1q(1.70852926535319*pi,1.0953669341897776*pi) q[16];
U1q(3.638440452263722*pi,0.5083005210820666*pi) q[17];
U1q(3.5819138667735*pi,1.6587845697180614*pi) q[18];
U1q(3.665597285358797*pi,0.5736156005042954*pi) q[19];
U1q(1.19281048149815*pi,0.7176240279450132*pi) q[20];
U1q(0.556580372180893*pi,1.1751660315967278*pi) q[21];
U1q(1.50002781976816*pi,0.6895566737933361*pi) q[22];
U1q(1.33379927334924*pi,1.91482513735096*pi) q[23];
U1q(0.12454728141848*pi,0.03207660400281154*pi) q[24];
U1q(1.64112623085276*pi,1.5766983926720268*pi) q[25];
U1q(3.931070347679864*pi,0.44078499575869*pi) q[26];
U1q(1.55202936680386*pi,1.8204554741749082*pi) q[27];
U1q(3.470156233068775*pi,1.4262604517072743*pi) q[28];
U1q(1.8865448713373*pi,0.9983040475846128*pi) q[29];
U1q(1.34290697133279*pi,0.08720572341545685*pi) q[30];
U1q(3.540749159772309*pi,0.6128535616209003*pi) q[31];
U1q(3.754636349580789*pi,1.795783620483704*pi) q[32];
U1q(3.299516716759498*pi,1.6777815898507424*pi) q[33];
U1q(3.274557760349194*pi,0.7250803120246756*pi) q[34];
U1q(1.63349845081272*pi,0.01952541075644798*pi) q[35];
U1q(3.161630424470061*pi,1.0387108020264257*pi) q[36];
U1q(0.134503987783652*pi,1.6544681859299235*pi) q[37];
U1q(3.4352788082874097*pi,0.9394454827473097*pi) q[38];
U1q(1.53238182137329*pi,1.3019349857144977*pi) q[39];
RZZ(0.5*pi) q[0],q[31];
RZZ(0.5*pi) q[16],q[1];
RZZ(0.5*pi) q[2],q[36];
RZZ(0.5*pi) q[18],q[3];
RZZ(0.5*pi) q[4],q[35];
RZZ(0.5*pi) q[5],q[10];
RZZ(0.5*pi) q[37],q[6];
RZZ(0.5*pi) q[28],q[7];
RZZ(0.5*pi) q[26],q[8];
RZZ(0.5*pi) q[9],q[17];
RZZ(0.5*pi) q[39],q[11];
RZZ(0.5*pi) q[19],q[12];
RZZ(0.5*pi) q[13],q[38];
RZZ(0.5*pi) q[14],q[32];
RZZ(0.5*pi) q[15],q[30];
RZZ(0.5*pi) q[20],q[22];
RZZ(0.5*pi) q[21],q[27];
RZZ(0.5*pi) q[23],q[24];
RZZ(0.5*pi) q[25],q[33];
RZZ(0.5*pi) q[34],q[29];
U1q(1.77899494688514*pi,1.041309416142819*pi) q[0];
U1q(1.28583453270203*pi,0.05015114756563133*pi) q[1];
U1q(3.4022752826702902*pi,0.9031550457543929*pi) q[2];
U1q(3.89983556362773*pi,1.8426031180611862*pi) q[3];
U1q(1.82424491579231*pi,1.2307153424832045*pi) q[4];
U1q(1.50296088839943*pi,0.8881460404087136*pi) q[5];
U1q(1.48337440160596*pi,0.790781932759701*pi) q[6];
U1q(1.70346990110298*pi,0.06629991525792933*pi) q[7];
U1q(0.513975753575252*pi,0.6189633587037733*pi) q[8];
U1q(3.697191241373253*pi,1.1399501055182997*pi) q[9];
U1q(0.76583497577309*pi,0.30117042047589315*pi) q[10];
U1q(1.61542460602914*pi,1.4298932619510727*pi) q[11];
U1q(1.11472793464202*pi,1.5525210039568966*pi) q[12];
U1q(1.76752993149969*pi,0.7150519031293641*pi) q[13];
U1q(0.410291732916435*pi,1.6527491869404596*pi) q[14];
U1q(1.57441112839458*pi,1.576932419727079*pi) q[15];
U1q(0.497461261507061*pi,0.5297053551922879*pi) q[16];
U1q(1.68180186438864*pi,0.5292888132079865*pi) q[17];
U1q(1.61620468329503*pi,1.0648576250967459*pi) q[18];
U1q(3.42236992545679*pi,1.4682706972995954*pi) q[19];
U1q(1.58422662328844*pi,1.0190851553958433*pi) q[20];
U1q(1.80936849443772*pi,1.899963443086918*pi) q[21];
U1q(0.42605383592255*pi,0.11519857335116646*pi) q[22];
U1q(0.156530682036207*pi,1.0865729649878197*pi) q[23];
U1q(0.423230809343907*pi,1.6680602945382215*pi) q[24];
U1q(3.759575728739708*pi,0.6208473308567433*pi) q[25];
U1q(3.451119584984739*pi,0.33302760626690997*pi) q[26];
U1q(3.635986448604597*pi,0.3357743370775621*pi) q[27];
U1q(0.288510139563535*pi,0.9027247097597844*pi) q[28];
U1q(1.77927047937432*pi,0.6623532973326043*pi) q[29];
U1q(1.73665491874039*pi,1.5418005337495568*pi) q[30];
U1q(3.501904164260296*pi,0.6014541444829407*pi) q[31];
U1q(1.69080822682686*pi,0.4334342866824743*pi) q[32];
U1q(1.56609702820313*pi,0.17220388267427245*pi) q[33];
U1q(3.406679120579779*pi,1.7375717304115956*pi) q[34];
U1q(1.56539974399811*pi,0.9020206404857785*pi) q[35];
U1q(1.48180251059856*pi,0.9867924431053003*pi) q[36];
U1q(0.191977562123973*pi,1.245785777166554*pi) q[37];
U1q(3.23495647419361*pi,0.4152657445154977*pi) q[38];
U1q(1.55673711449936*pi,1.5426865263364566*pi) q[39];
RZZ(0.5*pi) q[9],q[0];
RZZ(0.5*pi) q[1],q[37];
RZZ(0.5*pi) q[2],q[11];
RZZ(0.5*pi) q[12],q[3];
RZZ(0.5*pi) q[4],q[22];
RZZ(0.5*pi) q[5],q[34];
RZZ(0.5*pi) q[33],q[6];
RZZ(0.5*pi) q[10],q[7];
RZZ(0.5*pi) q[8],q[24];
RZZ(0.5*pi) q[13],q[31];
RZZ(0.5*pi) q[14],q[38];
RZZ(0.5*pi) q[15],q[28];
RZZ(0.5*pi) q[32],q[16];
RZZ(0.5*pi) q[18],q[17];
RZZ(0.5*pi) q[19],q[35];
RZZ(0.5*pi) q[30],q[20];
RZZ(0.5*pi) q[29],q[21];
RZZ(0.5*pi) q[23],q[25];
RZZ(0.5*pi) q[26],q[27];
RZZ(0.5*pi) q[39],q[36];
U1q(3.216338438477059*pi,0.46722441320350505*pi) q[0];
U1q(1.54579867933395*pi,0.10299820378889102*pi) q[1];
U1q(3.851224107088942*pi,1.993855027199623*pi) q[2];
U1q(3.379464961395477*pi,1.618408913951586*pi) q[3];
U1q(0.47020728784325*pi,0.6179629131946047*pi) q[4];
U1q(1.36105604401002*pi,1.1015217765719916*pi) q[5];
U1q(0.359927010712439*pi,1.547672812442261*pi) q[6];
U1q(1.77358877001758*pi,0.8790491987776896*pi) q[7];
U1q(0.502802042633838*pi,0.36668227838414325*pi) q[8];
U1q(3.609102684181656*pi,0.5685530861256005*pi) q[9];
U1q(1.43139782430361*pi,0.05715235227463378*pi) q[10];
U1q(3.259264397160213*pi,0.8568788814618231*pi) q[11];
U1q(3.44613309002986*pi,0.09497877316617753*pi) q[12];
U1q(0.255204112826944*pi,0.42784758602577355*pi) q[13];
U1q(1.80022357860723*pi,0.2672930060679697*pi) q[14];
U1q(0.85013524624472*pi,0.7255407238779195*pi) q[15];
U1q(3.469148767046911*pi,0.45511480733123744*pi) q[16];
U1q(1.73963849782464*pi,1.991763664732737*pi) q[17];
U1q(0.465727434894902*pi,1.7543625559202356*pi) q[18];
U1q(3.161032453200154*pi,0.4486783492317912*pi) q[19];
U1q(3.551137464679144*pi,1.541318004602715*pi) q[20];
U1q(3.537865868328276*pi,1.2837333950535865*pi) q[21];
U1q(0.61493188675014*pi,1.145848802799386*pi) q[22];
U1q(0.716700580470459*pi,0.22839801515932034*pi) q[23];
U1q(1.66162071742902*pi,0.24537025882731145*pi) q[24];
U1q(3.464206971751841*pi,1.4195828739489642*pi) q[25];
U1q(3.161320361730921*pi,1.9572319864235799*pi) q[26];
U1q(3.623463924761891*pi,1.696689919397092*pi) q[27];
U1q(0.655805621945422*pi,1.8539970889596935*pi) q[28];
U1q(0.362354217906048*pi,1.6859893648634952*pi) q[29];
U1q(3.528746208660126*pi,1.415898536387271*pi) q[30];
U1q(1.67363789569514*pi,1.2551875663388987*pi) q[31];
U1q(1.83896995622734*pi,0.2066374354645042*pi) q[32];
U1q(3.721953989425895*pi,1.517819451351003*pi) q[33];
U1q(3.597394193991854*pi,0.8583469173846595*pi) q[34];
U1q(1.42687387709268*pi,0.8931978013723367*pi) q[35];
U1q(0.498860735081501*pi,1.4673176015276708*pi) q[36];
U1q(0.339401956327089*pi,0.3311237388307733*pi) q[37];
U1q(1.84537359098613*pi,0.9553414071614945*pi) q[38];
U1q(1.65019257639347*pi,1.3985509701489072*pi) q[39];
RZZ(0.5*pi) q[20],q[0];
RZZ(0.5*pi) q[28],q[1];
RZZ(0.5*pi) q[2],q[39];
RZZ(0.5*pi) q[11],q[3];
RZZ(0.5*pi) q[4],q[32];
RZZ(0.5*pi) q[5],q[24];
RZZ(0.5*pi) q[10],q[6];
RZZ(0.5*pi) q[23],q[7];
RZZ(0.5*pi) q[18],q[8];
RZZ(0.5*pi) q[9],q[14];
RZZ(0.5*pi) q[12],q[33];
RZZ(0.5*pi) q[13],q[27];
RZZ(0.5*pi) q[15],q[16];
RZZ(0.5*pi) q[26],q[17];
RZZ(0.5*pi) q[19],q[22];
RZZ(0.5*pi) q[34],q[21];
RZZ(0.5*pi) q[25],q[37];
RZZ(0.5*pi) q[29],q[31];
RZZ(0.5*pi) q[30],q[36];
RZZ(0.5*pi) q[35],q[38];
U1q(3.528242786869513*pi,1.6946365353786925*pi) q[0];
U1q(0.951513236465593*pi,0.5247599447735212*pi) q[1];
U1q(3.288155765142432*pi,0.22671167191746822*pi) q[2];
U1q(3.418339915839295*pi,0.41208734210104936*pi) q[3];
U1q(0.435265887400797*pi,1.2076944868880952*pi) q[4];
U1q(0.165962877156357*pi,1.5986551774382214*pi) q[5];
U1q(0.421843493442058*pi,1.733353568620621*pi) q[6];
U1q(1.25435572650865*pi,0.32321944635447863*pi) q[7];
U1q(0.440333252307092*pi,1.8805884117613143*pi) q[8];
U1q(3.355588778158576*pi,0.3312155028967467*pi) q[9];
U1q(1.56597538314913*pi,1.9898906819997162*pi) q[10];
U1q(0.472951997989892*pi,0.8870074403988335*pi) q[11];
U1q(3.720437021063365*pi,1.6999463070221887*pi) q[12];
U1q(0.29867925188798*pi,0.4922425985117229*pi) q[13];
U1q(1.57688407146312*pi,1.4654218905682148*pi) q[14];
U1q(0.595298506907946*pi,0.0437712221048856*pi) q[15];
U1q(1.41180775384982*pi,1.882332627047897*pi) q[16];
U1q(1.38752147264021*pi,0.6627363269781945*pi) q[17];
U1q(0.0939591103955607*pi,0.18980512620961498*pi) q[18];
U1q(0.461572364630195*pi,0.06616537500763157*pi) q[19];
U1q(1.6511856991305*pi,1.9341814249793225*pi) q[20];
U1q(3.4119532031937743*pi,0.21723181813983228*pi) q[21];
U1q(0.838110210123081*pi,0.10391044773609615*pi) q[22];
U1q(0.70469761793426*pi,0.6055870595405803*pi) q[23];
U1q(1.31937132065088*pi,1.1201479202314402*pi) q[24];
U1q(1.59517791334852*pi,1.955141394595915*pi) q[25];
U1q(1.17949028033175*pi,0.9772556339912728*pi) q[26];
U1q(1.61401979441796*pi,1.9215596729599564*pi) q[27];
U1q(0.60373154296018*pi,1.0268518739222943*pi) q[28];
U1q(0.480557863542659*pi,0.6656242708732751*pi) q[29];
U1q(1.30706050084277*pi,0.6777536791933922*pi) q[30];
U1q(0.473552015574584*pi,0.8390954223396392*pi) q[31];
U1q(3.176351354491855*pi,1.3763839275751764*pi) q[32];
U1q(1.29698057457767*pi,1.3884624303478974*pi) q[33];
U1q(1.51157543691576*pi,1.683790037218459*pi) q[34];
U1q(0.517528476649153*pi,0.3777326194695476*pi) q[35];
U1q(0.719105319812198*pi,0.12904825934068054*pi) q[36];
U1q(0.310164565553696*pi,1.2182814436252833*pi) q[37];
U1q(0.661590652883406*pi,1.5422786946867544*pi) q[38];
U1q(1.45707272984702*pi,0.8246348269545072*pi) q[39];
rz(2.3053634646213075*pi) q[0];
rz(1.4752400552264788*pi) q[1];
rz(3.773288328082532*pi) q[2];
rz(3.5879126578989506*pi) q[3];
rz(2.792305513111905*pi) q[4];
rz(0.4013448225617786*pi) q[5];
rz(0.26664643137937905*pi) q[6];
rz(1.6767805536455214*pi) q[7];
rz(0.1194115882386857*pi) q[8];
rz(3.6687844971032533*pi) q[9];
rz(0.010109318000283807*pi) q[10];
rz(3.1129925596011665*pi) q[11];
rz(2.3000536929778113*pi) q[12];
rz(3.507757401488277*pi) q[13];
rz(2.534578109431785*pi) q[14];
rz(1.9562287778951144*pi) q[15];
rz(2.117667372952103*pi) q[16];
rz(3.3372636730218055*pi) q[17];
rz(1.810194873790385*pi) q[18];
rz(3.9338346249923686*pi) q[19];
rz(2.0658185750206775*pi) q[20];
rz(1.7827681818601677*pi) q[21];
rz(1.8960895522639039*pi) q[22];
rz(1.3944129404594197*pi) q[23];
rz(0.8798520797685598*pi) q[24];
rz(0.04485860540408493*pi) q[25];
rz(3.022744366008727*pi) q[26];
rz(0.07844032704004356*pi) q[27];
rz(2.9731481260777057*pi) q[28];
rz(3.334375729126725*pi) q[29];
rz(3.3222463208066078*pi) q[30];
rz(1.1609045776603608*pi) q[31];
rz(0.6236160724248236*pi) q[32];
rz(2.6115375696521026*pi) q[33];
rz(0.3162099627815409*pi) q[34];
rz(1.6222673805304524*pi) q[35];
rz(3.8709517406593195*pi) q[36];
rz(0.7817185563747167*pi) q[37];
rz(2.4577213053132456*pi) q[38];
rz(3.175365173045493*pi) q[39];
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