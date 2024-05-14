OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.507862633940902*pi,0.710700597022507*pi) q[0];
U1q(1.42487662148942*pi,0.20030981440036066*pi) q[1];
U1q(0.74223474421833*pi,0.966753855392848*pi) q[2];
U1q(0.560432185076442*pi,1.48182558767145*pi) q[3];
U1q(0.769936534240175*pi,1.12968254656887*pi) q[4];
U1q(0.402225867949466*pi,0.308655029203857*pi) q[5];
U1q(0.825639324582123*pi,0.667857440421731*pi) q[6];
U1q(3.193562990050731*pi,0.9492972684573505*pi) q[7];
U1q(1.35052085113536*pi,0.3731765943188794*pi) q[8];
U1q(0.161160636320979*pi,0.477000341668176*pi) q[9];
U1q(0.782264355089091*pi,0.0219727580215595*pi) q[10];
U1q(1.29027231494139*pi,0.13447065703197714*pi) q[11];
U1q(1.18488835891777*pi,0.561391060512324*pi) q[12];
U1q(0.544829183788057*pi,0.682053786802115*pi) q[13];
U1q(1.42838101824595*pi,0.19717446879173448*pi) q[14];
U1q(1.63317984014526*pi,0.4361481409834112*pi) q[15];
U1q(0.17132478289555*pi,0.7832116350059499*pi) q[16];
U1q(1.41989084434111*pi,1.5236381631710212*pi) q[17];
U1q(1.69197773089361*pi,0.19693238305955724*pi) q[18];
U1q(1.6577104197278*pi,0.05288565473718193*pi) q[19];
U1q(0.74902815157473*pi,0.176372799803875*pi) q[20];
U1q(1.78993015338003*pi,1.269388872701842*pi) q[21];
U1q(1.75872994457111*pi,0.39410593017591294*pi) q[22];
U1q(1.81873015660977*pi,0.29861871199009926*pi) q[23];
U1q(0.719250434420364*pi,1.425210854785719*pi) q[24];
U1q(0.341574765334441*pi,0.90187635259001*pi) q[25];
U1q(1.56458894597667*pi,1.3360746327821253*pi) q[26];
U1q(3.768074624473763*pi,0.4055170869797418*pi) q[27];
U1q(1.85206207696198*pi,1.6674886813871928*pi) q[28];
U1q(0.590910993517008*pi,0.859346789458337*pi) q[29];
U1q(0.562662718232832*pi,1.624864839123663*pi) q[30];
U1q(3.025773952926547*pi,1.5003024863620402*pi) q[31];
U1q(0.515243218266303*pi,0.0499087791738648*pi) q[32];
U1q(0.231913914086642*pi,1.596969118485057*pi) q[33];
U1q(1.46487346874406*pi,0.2703876900783162*pi) q[34];
U1q(1.33176396155301*pi,1.4102555392859988*pi) q[35];
U1q(0.229155569181754*pi,0.62746808442805*pi) q[36];
U1q(1.6735721188142*pi,0.35492177879782544*pi) q[37];
U1q(0.775250079761272*pi,0.44625426633405*pi) q[38];
U1q(0.838356580840466*pi,0.579369061727515*pi) q[39];
RZZ(0.5*pi) q[28],q[0];
RZZ(0.5*pi) q[1],q[10];
RZZ(0.5*pi) q[2],q[32];
RZZ(0.5*pi) q[3],q[8];
RZZ(0.5*pi) q[4],q[9];
RZZ(0.5*pi) q[16],q[5];
RZZ(0.5*pi) q[6],q[36];
RZZ(0.5*pi) q[7],q[12];
RZZ(0.5*pi) q[15],q[11];
RZZ(0.5*pi) q[13],q[31];
RZZ(0.5*pi) q[14],q[27];
RZZ(0.5*pi) q[18],q[17];
RZZ(0.5*pi) q[19],q[37];
RZZ(0.5*pi) q[20],q[21];
RZZ(0.5*pi) q[22],q[25];
RZZ(0.5*pi) q[23],q[39];
RZZ(0.5*pi) q[24],q[34];
RZZ(0.5*pi) q[26],q[38];
RZZ(0.5*pi) q[29],q[35];
RZZ(0.5*pi) q[30],q[33];
U1q(0.468810815414469*pi,1.653296020363082*pi) q[0];
U1q(0.56934972593053*pi,0.1067146134298107*pi) q[1];
U1q(0.627442177183225*pi,1.747923318656074*pi) q[2];
U1q(0.824833839876033*pi,1.760861120840073*pi) q[3];
U1q(0.047396040584691*pi,1.9507804788090302*pi) q[4];
U1q(0.308276856454229*pi,1.49075084682369*pi) q[5];
U1q(0.698989381946102*pi,0.52525158847104*pi) q[6];
U1q(0.0856005492752558*pi,1.7320662024458104*pi) q[7];
U1q(0.100719768005589*pi,1.5861326176685173*pi) q[8];
U1q(0.987950392933324*pi,0.6743881988842801*pi) q[9];
U1q(0.451360434435515*pi,0.4589746125169101*pi) q[10];
U1q(0.124643553693027*pi,1.195367743908017*pi) q[11];
U1q(0.370331452507216*pi,0.893452629391404*pi) q[12];
U1q(0.308787426980173*pi,0.315422267719681*pi) q[13];
U1q(0.338357227085847*pi,0.5544894115182841*pi) q[14];
U1q(0.737335735660755*pi,0.6515615853878516*pi) q[15];
U1q(0.59583703168254*pi,1.50877094109598*pi) q[16];
U1q(0.284229530221407*pi,1.3740374578938512*pi) q[17];
U1q(0.8494665992544*pi,0.19924150815615738*pi) q[18];
U1q(0.630375338621516*pi,1.1157027458689521*pi) q[19];
U1q(0.815310054392674*pi,0.16417693261779998*pi) q[20];
U1q(0.641969192521297*pi,0.2749641428769123*pi) q[21];
U1q(0.198096000828376*pi,1.4643892750509129*pi) q[22];
U1q(0.711822387697082*pi,0.6639792794029793*pi) q[23];
U1q(0.213914074512532*pi,0.7606800350810601*pi) q[24];
U1q(0.741785402460237*pi,0.20682362039440005*pi) q[25];
U1q(0.286549797522195*pi,0.1560389211083253*pi) q[26];
U1q(0.686142374233144*pi,1.4265283942507687*pi) q[27];
U1q(0.0504357986466858*pi,0.4972813215833032*pi) q[28];
U1q(0.252843930179331*pi,1.290583328147961*pi) q[29];
U1q(0.350987656799451*pi,1.60020666018099*pi) q[30];
U1q(0.511352560309845*pi,0.43262692621732013*pi) q[31];
U1q(0.650216991992566*pi,0.3008208298585999*pi) q[32];
U1q(0.654706811421325*pi,0.12708300147432006*pi) q[33];
U1q(0.613929788882632*pi,0.41818984396664627*pi) q[34];
U1q(0.602341932175063*pi,1.8287104034312591*pi) q[35];
U1q(0.319186745676579*pi,0.5943020090485902*pi) q[36];
U1q(0.174307000133978*pi,0.7977050621676454*pi) q[37];
U1q(0.802113481514642*pi,0.265959163203214*pi) q[38];
U1q(0.388714515223969*pi,1.96225576108053*pi) q[39];
RZZ(0.5*pi) q[33],q[0];
RZZ(0.5*pi) q[1],q[27];
RZZ(0.5*pi) q[2],q[21];
RZZ(0.5*pi) q[30],q[3];
RZZ(0.5*pi) q[4],q[6];
RZZ(0.5*pi) q[32],q[5];
RZZ(0.5*pi) q[7],q[8];
RZZ(0.5*pi) q[9],q[38];
RZZ(0.5*pi) q[19],q[10];
RZZ(0.5*pi) q[11],q[17];
RZZ(0.5*pi) q[12],q[31];
RZZ(0.5*pi) q[29],q[13];
RZZ(0.5*pi) q[14],q[22];
RZZ(0.5*pi) q[28],q[15];
RZZ(0.5*pi) q[16],q[26];
RZZ(0.5*pi) q[18],q[25];
RZZ(0.5*pi) q[20],q[39];
RZZ(0.5*pi) q[23],q[35];
RZZ(0.5*pi) q[24],q[37];
RZZ(0.5*pi) q[36],q[34];
U1q(0.313284687692519*pi,1.7261948143016301*pi) q[0];
U1q(0.559811282512637*pi,0.08060579805622092*pi) q[1];
U1q(0.732034451449859*pi,0.7124520544049*pi) q[2];
U1q(0.804501841599516*pi,1.9975801401668498*pi) q[3];
U1q(0.703927337227262*pi,1.1134659048608602*pi) q[4];
U1q(0.591854423604186*pi,0.09018188602319999*pi) q[5];
U1q(0.657682684152493*pi,0.8858537206135297*pi) q[6];
U1q(0.269280653317602*pi,0.05435454263975048*pi) q[7];
U1q(0.398268470265971*pi,1.4990056957682092*pi) q[8];
U1q(0.67133925962047*pi,0.2992697557880204*pi) q[9];
U1q(0.809999301750482*pi,1.5633128105297303*pi) q[10];
U1q(0.604462074685163*pi,1.0509525341564272*pi) q[11];
U1q(0.769739990332377*pi,0.09882941803861423*pi) q[12];
U1q(0.118794188252764*pi,1.7216308698841098*pi) q[13];
U1q(0.563512641048401*pi,1.536683858056775*pi) q[14];
U1q(0.654710788264181*pi,0.47138102153248074*pi) q[15];
U1q(0.120819162913048*pi,0.11729150329902005*pi) q[16];
U1q(0.146824490192384*pi,1.2543885154699108*pi) q[17];
U1q(0.750215898414263*pi,0.8541916945467065*pi) q[18];
U1q(0.481300657934195*pi,1.6210434831366713*pi) q[19];
U1q(0.169338152817328*pi,0.89787934716498*pi) q[20];
U1q(0.610607443810614*pi,0.2597394899635219*pi) q[21];
U1q(0.413044989614617*pi,0.33934258061908285*pi) q[22];
U1q(0.105977404512215*pi,1.3431048106325996*pi) q[23];
U1q(0.704906379300107*pi,0.4993171843196196*pi) q[24];
U1q(0.36374068952638*pi,0.16040354007860014*pi) q[25];
U1q(0.461770838800255*pi,1.5566018777391752*pi) q[26];
U1q(0.503994416448942*pi,0.3380295801942017*pi) q[27];
U1q(0.603695806197672*pi,1.5169402324328525*pi) q[28];
U1q(0.318522734075387*pi,1.08129670269488*pi) q[29];
U1q(0.831298774072944*pi,1.9246590897671396*pi) q[30];
U1q(0.51386450548362*pi,1.01369713858668*pi) q[31];
U1q(0.326126336069602*pi,0.27986929704962993*pi) q[32];
U1q(0.920843612052251*pi,1.9599476131513498*pi) q[33];
U1q(0.58384243193078*pi,1.8112265750452554*pi) q[34];
U1q(0.660680596049554*pi,0.19821055350519945*pi) q[35];
U1q(0.391141988380817*pi,0.5350508708189503*pi) q[36];
U1q(0.303627650068199*pi,0.7605517758626954*pi) q[37];
U1q(0.47631202951175*pi,0.56586827718486*pi) q[38];
U1q(0.556677792861495*pi,1.5401309680746902*pi) q[39];
RZZ(0.5*pi) q[6],q[0];
RZZ(0.5*pi) q[9],q[1];
RZZ(0.5*pi) q[2],q[12];
RZZ(0.5*pi) q[3],q[36];
RZZ(0.5*pi) q[4],q[30];
RZZ(0.5*pi) q[5],q[26];
RZZ(0.5*pi) q[7],q[23];
RZZ(0.5*pi) q[14],q[8];
RZZ(0.5*pi) q[24],q[10];
RZZ(0.5*pi) q[25],q[11];
RZZ(0.5*pi) q[13],q[19];
RZZ(0.5*pi) q[15],q[18];
RZZ(0.5*pi) q[16],q[37];
RZZ(0.5*pi) q[27],q[17];
RZZ(0.5*pi) q[20],q[29];
RZZ(0.5*pi) q[21],q[35];
RZZ(0.5*pi) q[22],q[38];
RZZ(0.5*pi) q[28],q[34];
RZZ(0.5*pi) q[32],q[31];
RZZ(0.5*pi) q[39],q[33];
U1q(0.587658037267043*pi,1.0503508025441501*pi) q[0];
U1q(0.610494725467414*pi,1.0895734508384702*pi) q[1];
U1q(0.152296435076176*pi,1.54212751547791*pi) q[2];
U1q(0.788518112662808*pi,0.50830488973335*pi) q[3];
U1q(0.28274869573716*pi,1.98654299322558*pi) q[4];
U1q(0.145240649199515*pi,0.4038965600965798*pi) q[5];
U1q(0.459904204220336*pi,1.5569168649978007*pi) q[6];
U1q(0.671789473456729*pi,1.88050316909354*pi) q[7];
U1q(0.265024604865646*pi,1.276644989062639*pi) q[8];
U1q(0.399754991505878*pi,1.7597496833335793*pi) q[9];
U1q(0.518408033667511*pi,1.4042871260174596*pi) q[10];
U1q(0.466076227240372*pi,1.8399239151163567*pi) q[11];
U1q(0.23468086719334*pi,1.3834818165754736*pi) q[12];
U1q(0.480347673939494*pi,0.27995646047696*pi) q[13];
U1q(0.611962776445675*pi,0.7480465852249143*pi) q[14];
U1q(0.683315202219992*pi,0.25532476587547137*pi) q[15];
U1q(0.498439938959817*pi,1.2117572979518396*pi) q[16];
U1q(0.563711265767179*pi,1.269654966477951*pi) q[17];
U1q(0.390991868885956*pi,0.8486617448080871*pi) q[18];
U1q(0.392034021051635*pi,1.3069125242301514*pi) q[19];
U1q(0.760908250977286*pi,1.3457703511401196*pi) q[20];
U1q(0.272193978520783*pi,0.5192840312626519*pi) q[21];
U1q(0.567109304533551*pi,1.6447047281155731*pi) q[22];
U1q(0.777826062356604*pi,1.9774694773476806*pi) q[23];
U1q(0.680168922755682*pi,0.2808987618694596*pi) q[24];
U1q(0.50032435827494*pi,0.9139876701123697*pi) q[25];
U1q(0.361440972336413*pi,1.2451004288629957*pi) q[26];
U1q(0.415260721441702*pi,1.6561422967542816*pi) q[27];
U1q(0.302322475488567*pi,0.8645412950384026*pi) q[28];
U1q(0.599679203650177*pi,0.12146176118432983*pi) q[29];
U1q(0.176992679755026*pi,1.1683930428035403*pi) q[30];
U1q(0.498252883297219*pi,0.005053509535129841*pi) q[31];
U1q(0.605092574122418*pi,1.5899497673443603*pi) q[32];
U1q(0.638195027854894*pi,0.18232537357624956*pi) q[33];
U1q(0.845243805540934*pi,1.3457526821442958*pi) q[34];
U1q(0.623488098927084*pi,1.8191673219416185*pi) q[35];
U1q(0.551863482190416*pi,0.11244145375768966*pi) q[36];
U1q(0.47526000653701*pi,0.03819194464920539*pi) q[37];
U1q(0.822278300421007*pi,0.026376504933950162*pi) q[38];
U1q(0.172248718375207*pi,0.3315156659651004*pi) q[39];
RZZ(0.5*pi) q[17],q[0];
RZZ(0.5*pi) q[2],q[1];
RZZ(0.5*pi) q[14],q[3];
RZZ(0.5*pi) q[4],q[33];
RZZ(0.5*pi) q[5],q[11];
RZZ(0.5*pi) q[19],q[6];
RZZ(0.5*pi) q[32],q[7];
RZZ(0.5*pi) q[12],q[8];
RZZ(0.5*pi) q[9],q[18];
RZZ(0.5*pi) q[10],q[25];
RZZ(0.5*pi) q[30],q[13];
RZZ(0.5*pi) q[15],q[34];
RZZ(0.5*pi) q[16],q[20];
RZZ(0.5*pi) q[23],q[21];
RZZ(0.5*pi) q[22],q[27];
RZZ(0.5*pi) q[24],q[28];
RZZ(0.5*pi) q[26],q[36];
RZZ(0.5*pi) q[29],q[31];
RZZ(0.5*pi) q[35],q[39];
RZZ(0.5*pi) q[37],q[38];
U1q(0.600837346024291*pi,1.4939014796122896*pi) q[0];
U1q(0.646601041905937*pi,0.2324887098317605*pi) q[1];
U1q(0.0833964010181983*pi,0.41771763860720945*pi) q[2];
U1q(0.690479871148621*pi,0.3684054989448402*pi) q[3];
U1q(0.846518203482533*pi,1.3768555825854696*pi) q[4];
U1q(0.216490758211433*pi,1.84532777351028*pi) q[5];
U1q(0.338981887260606*pi,0.10680579873790919*pi) q[6];
U1q(0.49208637566036*pi,0.7147155186175009*pi) q[7];
U1q(0.815055399001049*pi,1.8949918490054696*pi) q[8];
U1q(0.88621163221715*pi,0.9967075961465*pi) q[9];
U1q(0.161246690167243*pi,1.5445601462009009*pi) q[10];
U1q(0.658036925675433*pi,1.3876812670207173*pi) q[11];
U1q(0.232389352586397*pi,1.4634155050793538*pi) q[12];
U1q(0.616828340832661*pi,1.4286787896298296*pi) q[13];
U1q(0.325641072877722*pi,1.3030953123100648*pi) q[14];
U1q(0.526596356470871*pi,0.8058687150413117*pi) q[15];
U1q(0.855410887110065*pi,0.9276231776208803*pi) q[16];
U1q(0.426329804633455*pi,1.9660849599441015*pi) q[17];
U1q(0.147561590623739*pi,0.44231394284247827*pi) q[18];
U1q(0.687847500903832*pi,0.3305108031427313*pi) q[19];
U1q(0.577724161221518*pi,1.6873357808981897*pi) q[20];
U1q(0.651998988994945*pi,1.2771709696930333*pi) q[21];
U1q(0.337921594169273*pi,0.5707539433850632*pi) q[22];
U1q(0.505589134047274*pi,0.6265048612661506*pi) q[23];
U1q(0.480342878512066*pi,1.2317987008754994*pi) q[24];
U1q(0.441816611323829*pi,0.0022656083318999265*pi) q[25];
U1q(0.369274319594631*pi,1.381170054350486*pi) q[26];
U1q(0.60988790550362*pi,1.053394333981152*pi) q[27];
U1q(0.215362719030234*pi,0.465823459078603*pi) q[28];
U1q(0.382418777876454*pi,0.5056517724030698*pi) q[29];
U1q(0.537412302414735*pi,0.18011290208904018*pi) q[30];
U1q(0.238693167719986*pi,1.4520243189603494*pi) q[31];
U1q(0.309029323123728*pi,0.3644466094148502*pi) q[32];
U1q(0.42076201083922*pi,1.8214715192141195*pi) q[33];
U1q(0.964155191617681*pi,1.415333237971966*pi) q[34];
U1q(0.572222809368916*pi,1.6670648634572878*pi) q[35];
U1q(0.73430056915107*pi,0.1254546425748302*pi) q[36];
U1q(0.718425896703581*pi,1.0513591477911657*pi) q[37];
U1q(0.954027778689769*pi,0.8091558102164997*pi) q[38];
U1q(0.851376529241843*pi,1.6068353555639003*pi) q[39];
rz(2.3815443286724802*pi) q[0];
rz(1.1374452992332094*pi) q[1];
rz(2.4651653199475003*pi) q[2];
rz(1.2158264922779596*pi) q[3];
rz(3.0912800089235803*pi) q[4];
rz(2.0246099427157596*pi) q[5];
rz(2.6906472525189997*pi) q[6];
rz(2.8951128876151895*pi) q[7];
rz(3.5333334208188907*pi) q[8];
rz(2.9310660519155007*pi) q[9];
rz(1.7366956920625505*pi) q[10];
rz(2.335882225646513*pi) q[11];
rz(1.693955728237576*pi) q[12];
rz(0.7342129637200303*pi) q[13];
rz(3.872933243952936*pi) q[14];
rz(3.3246259453087887*pi) q[15];
rz(0.8845086444912198*pi) q[16];
rz(0.8283222273860691*pi) q[17];
rz(1.4758886643519418*pi) q[18];
rz(3.4441737368650287*pi) q[19];
rz(0.9967721042729796*pi) q[20];
rz(3.669574739046757*pi) q[21];
rz(0.339997442832507*pi) q[22];
rz(0.8660603036080605*pi) q[23];
rz(2.8427979995387993*pi) q[24];
rz(0.9420997308586401*pi) q[25];
rz(2.303228079425484*pi) q[26];
rz(1.6393890079673081*pi) q[27];
rz(1.1968182638035874*pi) q[28];
rz(0.47490432566872975*pi) q[29];
rz(0.06702943953735918*pi) q[30];
rz(0.24834285037255022*pi) q[31];
rz(2.5738149124101994*pi) q[32];
rz(0.9914330183076903*pi) q[33];
rz(3.8661348249231935*pi) q[34];
rz(3.0036591117513023*pi) q[35];
rz(1.16321626788989*pi) q[36];
rz(2.420120799014475*pi) q[37];
rz(0.6372834396025997*pi) q[38];
rz(2.281639438051*pi) q[39];
U1q(0.600837346024291*pi,0.875445808284763*pi) q[0];
U1q(0.646601041905937*pi,0.369934009064963*pi) q[1];
U1q(0.0833964010181983*pi,1.882882958554701*pi) q[2];
U1q(1.69047987114862*pi,0.584231991222794*pi) q[3];
U1q(1.84651820348253*pi,1.468135591509055*pi) q[4];
U1q(0.216490758211433*pi,0.86993771622604*pi) q[5];
U1q(0.338981887260606*pi,1.797453051256955*pi) q[6];
U1q(0.49208637566036*pi,0.609828406232687*pi) q[7];
U1q(1.81505539900105*pi,0.428325269824356*pi) q[8];
U1q(0.88621163221715*pi,0.92777364806209*pi) q[9];
U1q(1.16124669016724*pi,0.281255838263435*pi) q[10];
U1q(1.65803692567543*pi,0.723563492667221*pi) q[11];
U1q(0.232389352586397*pi,0.15737123331691*pi) q[12];
U1q(1.61682834083266*pi,1.16289175334986*pi) q[13];
U1q(0.325641072877722*pi,0.176028556263008*pi) q[14];
U1q(0.526596356470871*pi,1.130494660350148*pi) q[15];
U1q(1.85541088711007*pi,0.812131822112099*pi) q[16];
U1q(0.426329804633455*pi,1.794407187330175*pi) q[17];
U1q(0.147561590623739*pi,0.918202607194415*pi) q[18];
U1q(1.68784750090383*pi,0.774684540007758*pi) q[19];
U1q(0.577724161221518*pi,1.684107885171173*pi) q[20];
U1q(0.651998988994945*pi,1.9467457087398272*pi) q[21];
U1q(1.33792159416927*pi,1.910751386217568*pi) q[22];
U1q(1.50558913404727*pi,0.492565164874201*pi) q[23];
U1q(0.480342878512066*pi,1.074596700414233*pi) q[24];
U1q(1.44181661132383*pi,1.9443653391905467*pi) q[25];
U1q(0.369274319594631*pi,0.684398133775967*pi) q[26];
U1q(0.60988790550362*pi,1.692783341948462*pi) q[27];
U1q(0.215362719030234*pi,0.662641722882192*pi) q[28];
U1q(1.38241877787645*pi,1.9805560980717958*pi) q[29];
U1q(0.537412302414735*pi,1.247142341626402*pi) q[30];
U1q(0.238693167719986*pi,0.7003671693329001*pi) q[31];
U1q(1.30902932312373*pi,1.9382615218250872*pi) q[32];
U1q(1.42076201083922*pi,1.812904537521816*pi) q[33];
U1q(3.964155191617682*pi,0.281468062895159*pi) q[34];
U1q(1.57222280936892*pi,1.67072397520854*pi) q[35];
U1q(0.73430056915107*pi,0.288670910464727*pi) q[36];
U1q(1.71842589670358*pi,0.471479946805598*pi) q[37];
U1q(1.95402777868977*pi,0.446439249819098*pi) q[38];
U1q(0.851376529241843*pi,0.88847479361494*pi) q[39];
RZZ(0.5*pi) q[17],q[0];
RZZ(0.5*pi) q[2],q[1];
RZZ(0.5*pi) q[14],q[3];
RZZ(0.5*pi) q[4],q[33];
RZZ(0.5*pi) q[5],q[11];
RZZ(0.5*pi) q[19],q[6];
RZZ(0.5*pi) q[32],q[7];
RZZ(0.5*pi) q[12],q[8];
RZZ(0.5*pi) q[9],q[18];
RZZ(0.5*pi) q[10],q[25];
RZZ(0.5*pi) q[30],q[13];
RZZ(0.5*pi) q[15],q[34];
RZZ(0.5*pi) q[16],q[20];
RZZ(0.5*pi) q[23],q[21];
RZZ(0.5*pi) q[22],q[27];
RZZ(0.5*pi) q[24],q[28];
RZZ(0.5*pi) q[26],q[36];
RZZ(0.5*pi) q[29],q[31];
RZZ(0.5*pi) q[35],q[39];
RZZ(0.5*pi) q[37],q[38];
U1q(0.587658037267043*pi,1.4318951312166281*pi) q[0];
U1q(1.61049472546741*pi,0.227018750071681*pi) q[1];
U1q(0.152296435076176*pi,0.007292835425410038*pi) q[2];
U1q(3.211481887337192*pi,1.4443326004342798*pi) q[3];
U1q(1.28274869573716*pi,1.8584481808689453*pi) q[4];
U1q(1.14524064919952*pi,1.428506502812344*pi) q[5];
U1q(1.45990420422034*pi,1.2475641175168382*pi) q[6];
U1q(0.671789473456729*pi,1.77561605670873*pi) q[7];
U1q(3.734975395134354*pi,0.04667212976718227*pi) q[8];
U1q(1.39975499150588*pi,0.69081573524912*pi) q[9];
U1q(1.51840803366751*pi,1.421528858446858*pi) q[10];
U1q(3.533923772759627*pi,1.2713208445715773*pi) q[11];
U1q(1.23468086719334*pi,0.07743754481303*pi) q[12];
U1q(3.480347673939494*pi,0.3116140825027215*pi) q[13];
U1q(0.611962776445675*pi,1.62097982917785*pi) q[14];
U1q(1.68331520221999*pi,0.5799507111842701*pi) q[15];
U1q(1.49843993895982*pi,0.5279977017811381*pi) q[16];
U1q(0.563711265767179*pi,0.09797719386402992*pi) q[17];
U1q(0.390991868885956*pi,0.32455040916003*pi) q[18];
U1q(3.392034021051635*pi,1.7982828189203326*pi) q[19];
U1q(0.760908250977286*pi,1.342542455413108*pi) q[20];
U1q(1.27219397852078*pi,1.1888587703094502*pi) q[21];
U1q(1.56710930453355*pi,0.836800601487059*pi) q[22];
U1q(3.222173937643396*pi,0.1416005487926727*pi) q[23];
U1q(0.680168922755682*pi,1.1236967614082398*pi) q[24];
U1q(1.50032435827494*pi,0.03264327741008444*pi) q[25];
U1q(1.36144097233641*pi,1.5483285082884701*pi) q[26];
U1q(0.415260721441702*pi,0.29553130472159994*pi) q[27];
U1q(1.30232247548857*pi,0.0613595588419896*pi) q[28];
U1q(3.599679203650177*pi,0.36474610929053136*pi) q[29];
U1q(0.176992679755026*pi,0.2354224823408999*pi) q[30];
U1q(1.49825288329722*pi,1.25339635990768*pi) q[31];
U1q(1.60509257412242*pi,0.7127583638955753*pi) q[32];
U1q(3.638195027854894*pi,0.45205068315969266*pi) q[33];
U1q(3.154756194459066*pi,1.3510486187228208*pi) q[34];
U1q(3.3765119010729148*pi,0.5186215167242147*pi) q[35];
U1q(0.551863482190416*pi,1.27565772164759*pi) q[36];
U1q(1.47526000653701*pi,0.4846471499475614*pi) q[37];
U1q(1.82227830042101*pi,0.22921855510164457*pi) q[38];
U1q(1.17224871837521*pi,1.6131551040161103*pi) q[39];
RZZ(0.5*pi) q[6],q[0];
RZZ(0.5*pi) q[9],q[1];
RZZ(0.5*pi) q[2],q[12];
RZZ(0.5*pi) q[3],q[36];
RZZ(0.5*pi) q[4],q[30];
RZZ(0.5*pi) q[5],q[26];
RZZ(0.5*pi) q[7],q[23];
RZZ(0.5*pi) q[14],q[8];
RZZ(0.5*pi) q[24],q[10];
RZZ(0.5*pi) q[25],q[11];
RZZ(0.5*pi) q[13],q[19];
RZZ(0.5*pi) q[15],q[18];
RZZ(0.5*pi) q[16],q[37];
RZZ(0.5*pi) q[27],q[17];
RZZ(0.5*pi) q[20],q[29];
RZZ(0.5*pi) q[21],q[35];
RZZ(0.5*pi) q[22],q[38];
RZZ(0.5*pi) q[28],q[34];
RZZ(0.5*pi) q[32],q[31];
RZZ(0.5*pi) q[39],q[33];
U1q(1.31328468769252*pi,0.10773914297410991*pi) q[0];
U1q(3.559811282512637*pi,0.23598640285393946*pi) q[1];
U1q(3.732034451449859*pi,1.1776173743523901*pi) q[2];
U1q(3.195498158400484*pi,1.95505735000078*pi) q[3];
U1q(0.703927337227262*pi,1.9853710925042254*pi) q[4];
U1q(3.408145576395813*pi,1.7422211768857192*pi) q[5];
U1q(1.65768268415249*pi,1.918627261901099*pi) q[6];
U1q(1.2692806533176*pi,1.949467430254933*pi) q[7];
U1q(3.601731529734029*pi,1.8243114230616202*pi) q[8];
U1q(1.67133925962047*pi,1.1512956627946807*pi) q[9];
U1q(3.809999301750482*pi,0.5805545429591228*pi) q[10];
U1q(3.3955379253148372*pi,0.06029222553149749*pi) q[11];
U1q(3.769739990332377*pi,1.3620899433498836*pi) q[12];
U1q(3.118794188252764*pi,0.7532884919098626*pi) q[13];
U1q(0.563512641048401*pi,0.4096171020097099*pi) q[14];
U1q(1.65471078826418*pi,0.36389445552726607*pi) q[15];
U1q(0.120819162913048*pi,0.43353190712831924*pi) q[16];
U1q(0.146824490192384*pi,1.08271074285598*pi) q[17];
U1q(0.750215898414263*pi,0.330080358898649*pi) q[18];
U1q(1.4813006579342*pi,1.1124137778268546*pi) q[19];
U1q(1.16933815281733*pi,1.89465145143796*pi) q[20];
U1q(1.61060744381061*pi,0.4484033116085815*pi) q[21];
U1q(1.41304498961462*pi,0.531438453990571*pi) q[22];
U1q(3.894022595487785*pi,0.7759652155077437*pi) q[23];
U1q(3.704906379300107*pi,0.34211518385839*pi) q[24];
U1q(0.36374068952638*pi,1.2790591473763144*pi) q[25];
U1q(1.46177083880026*pi,0.23682705941228632*pi) q[26];
U1q(1.50399441644894*pi,1.9774185881615098*pi) q[27];
U1q(3.3963041938023277*pi,1.4089606214475332*pi) q[28];
U1q(0.318522734075387*pi,0.32458105080108224*pi) q[29];
U1q(0.831298774072944*pi,1.9916885293045001*pi) q[30];
U1q(3.48613549451638*pi,1.2447527308561268*pi) q[31];
U1q(1.3261263360696*pi,1.4026778936008482*pi) q[32];
U1q(0.920843612052251*pi,0.22967292273479245*pi) q[33];
U1q(1.58384243193078*pi,0.8855747258218711*pi) q[34];
U1q(1.66068059604955*pi,0.13957828516062687*pi) q[35];
U1q(1.39114198838082*pi,1.6982671387088404*pi) q[36];
U1q(0.303627650068199*pi,0.20700698116105354*pi) q[37];
U1q(0.47631202951175*pi,1.7687103273525566*pi) q[38];
U1q(3.443322207138505*pi,1.4045398019065143*pi) q[39];
RZZ(0.5*pi) q[33],q[0];
RZZ(0.5*pi) q[1],q[27];
RZZ(0.5*pi) q[2],q[21];
RZZ(0.5*pi) q[30],q[3];
RZZ(0.5*pi) q[4],q[6];
RZZ(0.5*pi) q[32],q[5];
RZZ(0.5*pi) q[7],q[8];
RZZ(0.5*pi) q[9],q[38];
RZZ(0.5*pi) q[19],q[10];
RZZ(0.5*pi) q[11],q[17];
RZZ(0.5*pi) q[12],q[31];
RZZ(0.5*pi) q[29],q[13];
RZZ(0.5*pi) q[14],q[22];
RZZ(0.5*pi) q[28],q[15];
RZZ(0.5*pi) q[16],q[26];
RZZ(0.5*pi) q[18],q[25];
RZZ(0.5*pi) q[20],q[39];
RZZ(0.5*pi) q[23],q[35];
RZZ(0.5*pi) q[24],q[37];
RZZ(0.5*pi) q[36],q[34];
U1q(1.46881081541447*pi,0.18063793691266072*pi) q[0];
U1q(1.56934972593053*pi,0.26209521822752846*pi) q[1];
U1q(1.62744217718323*pi,0.1421461101012138*pi) q[2];
U1q(1.82483383987603*pi,1.191776369327561*pi) q[3];
U1q(1.04739604058469*pi,1.8226856664523954*pi) q[4];
U1q(3.691723143545771*pi,1.3416522160852353*pi) q[5];
U1q(0.698989381946102*pi,0.5580251297586005*pi) q[6];
U1q(1.08560054927526*pi,1.271755770448867*pi) q[7];
U1q(3.100719768005589*pi,1.7371845011613076*pi) q[8];
U1q(1.98795039293332*pi,0.5264141058909511*pi) q[9];
U1q(1.45136043443552*pi,0.6848927409719385*pi) q[10];
U1q(1.12464355369303*pi,0.915877015779913*pi) q[11];
U1q(0.370331452507216*pi,0.15671315470267366*pi) q[12];
U1q(3.691212573019827*pi,1.1594970940742872*pi) q[13];
U1q(0.338357227085847*pi,1.4274226554712302*pi) q[14];
U1q(1.73733573566076*pi,0.544075019382646*pi) q[15];
U1q(1.59583703168254*pi,0.8250113449252789*pi) q[16];
U1q(1.28422953022141*pi,1.2023596852799203*pi) q[17];
U1q(0.8494665992544*pi,0.6751301725081*pi) q[18];
U1q(3.369624661378484*pi,1.6177545150945785*pi) q[19];
U1q(1.81531005439267*pi,1.628353865985145*pi) q[20];
U1q(1.6419691925213*pi,0.4636279645219705*pi) q[21];
U1q(1.19809600082838*pi,1.4063917595587436*pi) q[22];
U1q(3.711822387697082*pi,0.4550907467373708*pi) q[23];
U1q(3.786085925487467*pi,1.0807523330969522*pi) q[24];
U1q(1.74178540246024*pi,0.32547922769211457*pi) q[25];
U1q(3.286549797522195*pi,1.8362641027814464*pi) q[26];
U1q(1.68614237423314*pi,0.8889197741049442*pi) q[27];
U1q(1.05043579864669*pi,1.4286195322970858*pi) q[28];
U1q(0.252843930179331*pi,0.5338676762541557*pi) q[29];
U1q(0.350987656799451*pi,1.66723609971835*pi) q[30];
U1q(3.488647439690155*pi,1.8258229432254969*pi) q[31];
U1q(1.65021699199257*pi,0.3817263607918835*pi) q[32];
U1q(1.65470681142133*pi,0.3968083110577627*pi) q[33];
U1q(0.613929788882632*pi,1.4925379947432662*pi) q[34];
U1q(1.60234193217506*pi,1.7700781350866786*pi) q[35];
U1q(3.680813254323421*pi,0.6390160004791978*pi) q[36];
U1q(3.174307000133977*pi,0.24416026746599373*pi) q[37];
U1q(1.80211348151464*pi,0.4688012133709063*pi) q[38];
U1q(3.611285484776031*pi,1.9824150089006753*pi) q[39];
RZZ(0.5*pi) q[28],q[0];
RZZ(0.5*pi) q[1],q[10];
RZZ(0.5*pi) q[2],q[32];
RZZ(0.5*pi) q[3],q[8];
RZZ(0.5*pi) q[4],q[9];
RZZ(0.5*pi) q[16],q[5];
RZZ(0.5*pi) q[6],q[36];
RZZ(0.5*pi) q[7],q[12];
RZZ(0.5*pi) q[15],q[11];
RZZ(0.5*pi) q[13],q[31];
RZZ(0.5*pi) q[14],q[27];
RZZ(0.5*pi) q[18],q[17];
RZZ(0.5*pi) q[19],q[37];
RZZ(0.5*pi) q[20],q[21];
RZZ(0.5*pi) q[22],q[25];
RZZ(0.5*pi) q[23],q[39];
RZZ(0.5*pi) q[24],q[34];
RZZ(0.5*pi) q[26],q[38];
RZZ(0.5*pi) q[29],q[35];
RZZ(0.5*pi) q[30],q[33];
U1q(0.507862633940902*pi,0.23804251357209072*pi) q[0];
U1q(1.42487662148942*pi,0.16850001725697838*pi) q[1];
U1q(0.74223474421833*pi,0.3609766468379938*pi) q[2];
U1q(0.560432185076442*pi,1.9127408361589406*pi) q[3];
U1q(3.769936534240176*pi,1.6437835986925533*pi) q[4];
U1q(1.40222586794947*pi,0.5237480337050642*pi) q[5];
U1q(0.825639324582123*pi,0.7006309817092908*pi) q[6];
U1q(0.193562990050731*pi,1.4889868364604069*pi) q[7];
U1q(0.350520851135364*pi,1.5242284778116675*pi) q[8];
U1q(1.16116063632098*pi,1.72380196310705*pi) q[9];
U1q(0.782264355089091*pi,0.2478908864765783*pi) q[10];
U1q(0.290272314941388*pi,1.8549799289038749*pi) q[11];
U1q(0.184888358917774*pi,0.8246515858235837*pi) q[12];
U1q(1.54482918378806*pi,0.7928655749918483*pi) q[13];
U1q(0.42838101824595*pi,0.0701077127446803*pi) q[14];
U1q(1.63317984014526*pi,0.7594884637870909*pi) q[15];
U1q(3.17132478289555*pi,0.5505706510153043*pi) q[16];
U1q(1.41989084434111*pi,1.0527589800027473*pi) q[17];
U1q(0.691977730893613*pi,0.6728210474115*pi) q[18];
U1q(1.6577104197278*pi,0.6805716062263518*pi) q[19];
U1q(0.74902815157473*pi,1.640549733171225*pi) q[20];
U1q(1.78993015338003*pi,1.4692032346970416*pi) q[21];
U1q(0.758729944571108*pi,0.33610841468374364*pi) q[22];
U1q(0.818730156609767*pi,0.08973017932449157*pi) q[23];
U1q(3.719250434420365*pi,1.416221513392287*pi) q[24];
U1q(1.34157476533444*pi,1.6304264954965064*pi) q[25];
U1q(1.56458894597667*pi,1.6562283911076499*pi) q[26];
U1q(0.768074624473763*pi,0.8679084668339243*pi) q[27];
U1q(0.852062076961977*pi,1.5988268921009658*pi) q[28];
U1q(0.590910993517008*pi,0.10263113756453546*pi) q[29];
U1q(0.562662718232832*pi,1.69189427866102*pi) q[30];
U1q(3.025773952926547*pi,0.7581473830807699*pi) q[31];
U1q(0.515243218266303*pi,1.1308143101071533*pi) q[32];
U1q(1.23191391408664*pi,0.9269221940470311*pi) q[33];
U1q(0.46487346874406*pi,1.344735840854935*pi) q[34];
U1q(1.33176396155301*pi,0.18853299923193445*pi) q[35];
U1q(3.2291555691817537*pi,1.605849925099741*pi) q[36];
U1q(3.6735721188141968*pi,1.6869435508358093*pi) q[37];
U1q(1.77525007976127*pi,1.28850611024007*pi) q[38];
U1q(3.838356580840466*pi,0.3653017082536998*pi) q[39];
rz(3.7619574864279093*pi) q[0];
rz(1.8314999827430216*pi) q[1];
rz(1.6390233531620062*pi) q[2];
rz(0.08725916384105936*pi) q[3];
rz(0.3562164013074467*pi) q[4];
rz(1.4762519662949358*pi) q[5];
rz(1.2993690182907092*pi) q[6];
rz(2.511013163539593*pi) q[7];
rz(2.4757715221883325*pi) q[8];
rz(0.27619803689294997*pi) q[9];
rz(3.7521091135234217*pi) q[10];
rz(0.14502007109612516*pi) q[11];
rz(3.1753484141764163*pi) q[12];
rz(1.2071344250081517*pi) q[13];
rz(3.9298922872553197*pi) q[14];
rz(1.240511536212909*pi) q[15];
rz(1.4494293489846957*pi) q[16];
rz(0.9472410199972527*pi) q[17];
rz(1.3271789525885*pi) q[18];
rz(1.3194283937736482*pi) q[19];
rz(2.359450266828775*pi) q[20];
rz(2.5307967653029584*pi) q[21];
rz(1.6638915853162564*pi) q[22];
rz(3.9102698206755084*pi) q[23];
rz(2.583778486607713*pi) q[24];
rz(2.3695735045034936*pi) q[25];
rz(2.34377160889235*pi) q[26];
rz(1.1320915331660757*pi) q[27];
rz(2.401173107899034*pi) q[28];
rz(3.8973688624354645*pi) q[29];
rz(0.30810572133897995*pi) q[30];
rz(3.24185261691923*pi) q[31];
rz(2.8691856898928467*pi) q[32];
rz(3.073077805952969*pi) q[33];
rz(2.655264159145065*pi) q[34];
rz(3.8114670007680655*pi) q[35];
rz(2.394150074900259*pi) q[36];
rz(0.31305644916419073*pi) q[37];
rz(2.71149388975993*pi) q[38];
rz(3.6346982917463*pi) q[39];
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