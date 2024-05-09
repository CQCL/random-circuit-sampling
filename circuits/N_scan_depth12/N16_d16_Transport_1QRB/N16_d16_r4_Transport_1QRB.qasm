OPENQASM 2.0;
include "hqslib1.inc";

qreg q[16];
creg c[16];
rz(1.8161494107717295*pi) q[0];
rz(1.7999874252933328*pi) q[1];
rz(3.784554276069317*pi) q[2];
rz(0.0571785756474552*pi) q[3];
rz(1.7139125561681272*pi) q[4];
rz(0.424780709018153*pi) q[5];
rz(0.650475843214729*pi) q[6];
rz(3.749925443909569*pi) q[7];
rz(0.512557639119989*pi) q[8];
rz(2.14072873601615*pi) q[9];
rz(1.3063244036471*pi) q[10];
rz(0.424386195484955*pi) q[11];
rz(0.6046786038853825*pi) q[12];
rz(0.765453421432527*pi) q[13];
rz(2.58028817150736*pi) q[14];
rz(1.294082951948704*pi) q[15];
U1q(1.4252348684623*pi,1.27054673339624*pi) q[0];
U1q(1.13400278879448*pi,0.91549917306688*pi) q[1];
U1q(0.702784467970253*pi,0.512739768435962*pi) q[2];
U1q(0.760674654456764*pi,0.71861959501156*pi) q[3];
U1q(1.19196399810358*pi,1.05708585212172*pi) q[4];
U1q(0.472339288611271*pi,0.601169048565094*pi) q[5];
U1q(0.299750307388434*pi,0.196378672507101*pi) q[6];
U1q(0.232470081091355*pi,0.805737410707787*pi) q[7];
U1q(0.268901258487822*pi,1.1871215839501*pi) q[8];
U1q(0.749639510661222*pi,1.598598664143499*pi) q[9];
U1q(0.611111282450071*pi,0.643305648119952*pi) q[10];
U1q(0.379890580280353*pi,1.8374690948587231*pi) q[11];
U1q(1.48065355507665*pi,0.560796007452636*pi) q[12];
U1q(0.902751217765134*pi,0.787879697706335*pi) q[13];
U1q(0.815407967785841*pi,1.817177925041927*pi) q[14];
U1q(1.66135246082823*pi,0.593196141785575*pi) q[15];
RZZ(0.0*pi) q[0],q[8];
RZZ(0.0*pi) q[4],q[1];
RZZ(0.0*pi) q[13],q[2];
RZZ(0.0*pi) q[3],q[6];
RZZ(0.0*pi) q[5],q[7];
RZZ(0.0*pi) q[9],q[14];
RZZ(0.0*pi) q[10],q[12];
RZZ(0.0*pi) q[11],q[15];
rz(1.2633924847195*pi) q[0];
rz(0.94693265254605*pi) q[1];
rz(0.186262532482621*pi) q[2];
rz(2.59468930854457*pi) q[3];
rz(0.726942407346029*pi) q[4];
rz(1.68228179270388*pi) q[5];
rz(1.08149482090263*pi) q[6];
rz(0.132794134003725*pi) q[7];
rz(3.57150125940304*pi) q[8];
rz(3.552355224243219*pi) q[9];
rz(3.555280566677796*pi) q[10];
rz(1.10048927186386*pi) q[11];
rz(0.00494296128545335*pi) q[12];
rz(3.967349149361188*pi) q[13];
rz(1.00860896635166*pi) q[14];
rz(0.954291167696411*pi) q[15];
U1q(0.922982538890745*pi,0.591682031816991*pi) q[0];
U1q(0.459369978059931*pi,1.4221143254468*pi) q[1];
U1q(0.0965980990653588*pi,1.677600673169957*pi) q[2];
U1q(0.918693311256827*pi,0.0319222122745008*pi) q[3];
U1q(0.684526460041539*pi,0.684762067663075*pi) q[4];
U1q(0.723712050356876*pi,1.17888537761664*pi) q[5];
U1q(0.339119010566266*pi,1.9179027512273197*pi) q[6];
U1q(0.861105402607239*pi,0.510735192157275*pi) q[7];
U1q(0.315670926177782*pi,1.873937177610716*pi) q[8];
U1q(0.267235929986932*pi,1.105004659797632*pi) q[9];
U1q(0.405564161892515*pi,0.451865109329948*pi) q[10];
U1q(0.420485240033741*pi,0.226440437679682*pi) q[11];
U1q(0.798805983581882*pi,1.880797738674282*pi) q[12];
U1q(0.113890965241655*pi,1.403420446727044*pi) q[13];
U1q(0.298559141775186*pi,1.19578530566748*pi) q[14];
U1q(0.475978037490651*pi,0.280909077211066*pi) q[15];
RZZ(0.0*pi) q[5],q[0];
RZZ(0.0*pi) q[10],q[1];
RZZ(0.0*pi) q[2],q[6];
RZZ(0.0*pi) q[3],q[14];
RZZ(0.0*pi) q[4],q[11];
RZZ(0.0*pi) q[7],q[12];
RZZ(0.0*pi) q[15],q[8];
RZZ(0.0*pi) q[9],q[13];
rz(1.69648891431927*pi) q[0];
rz(1.03265114674597*pi) q[1];
rz(0.518753440683307*pi) q[2];
rz(1.22019117026659*pi) q[3];
rz(1.09532131177579*pi) q[4];
rz(2.48096945084938*pi) q[5];
rz(3.443987187195838*pi) q[6];
rz(0.2224833434975*pi) q[7];
rz(0.0121424110233255*pi) q[8];
rz(0.690535870282257*pi) q[9];
rz(0.753554436865069*pi) q[10];
rz(0.991946608854004*pi) q[11];
rz(1.45091214334057*pi) q[12];
rz(3.859794899720466*pi) q[13];
rz(2.8234344355693697*pi) q[14];
rz(0.818024558735597*pi) q[15];
U1q(0.64598701268027*pi,0.695281956704472*pi) q[0];
U1q(0.545380224941479*pi,1.79158491869009*pi) q[1];
U1q(0.680311000546722*pi,0.0653356523336788*pi) q[2];
U1q(0.89501551628858*pi,0.831059457751143*pi) q[3];
U1q(0.571850638147241*pi,0.494495683515466*pi) q[4];
U1q(0.670475993783814*pi,1.096932640972283*pi) q[5];
U1q(0.64031923208901*pi,1.585707557615425*pi) q[6];
U1q(0.22485993006111*pi,0.0712071555071184*pi) q[7];
U1q(0.638611630621321*pi,1.871720338083896*pi) q[8];
U1q(0.616752602015532*pi,0.345271330824534*pi) q[9];
U1q(0.820630724851246*pi,0.438561625034647*pi) q[10];
U1q(0.40285927181099*pi,1.25435843197678*pi) q[11];
U1q(0.110554362783316*pi,0.109241155815371*pi) q[12];
U1q(0.265112228932222*pi,1.252756964297304*pi) q[13];
U1q(0.659586325101376*pi,1.259076561647424*pi) q[14];
U1q(0.326401625735405*pi,1.584353654406319*pi) q[15];
RZZ(0.0*pi) q[3],q[0];
RZZ(0.0*pi) q[11],q[1];
RZZ(0.0*pi) q[10],q[2];
RZZ(0.0*pi) q[4],q[8];
RZZ(0.0*pi) q[9],q[5];
RZZ(0.0*pi) q[6],q[12];
RZZ(0.0*pi) q[15],q[7];
RZZ(0.0*pi) q[13],q[14];
rz(0.0739579955433239*pi) q[0];
rz(3.545854136049788*pi) q[1];
rz(2.89928554182554*pi) q[2];
rz(3.964089021131447*pi) q[3];
rz(0.529389701496532*pi) q[4];
rz(2.35307456835965*pi) q[5];
rz(2.27757139508833*pi) q[6];
rz(2.72033429617003*pi) q[7];
rz(2.60278398108738*pi) q[8];
rz(1.98899193010678*pi) q[9];
rz(3.136618714552422*pi) q[10];
rz(3.86476711199592*pi) q[11];
rz(0.403363812001461*pi) q[12];
rz(1.37066198282814*pi) q[13];
rz(2.54199324596*pi) q[14];
rz(0.151387845957947*pi) q[15];
U1q(0.951020011900377*pi,1.912272663906685*pi) q[0];
U1q(0.685801949178157*pi,0.200300344978663*pi) q[1];
U1q(0.649920728670684*pi,1.456244292240189*pi) q[2];
U1q(0.447466696054043*pi,1.488894356980015*pi) q[3];
U1q(0.856300889935202*pi,0.929021125728361*pi) q[4];
U1q(0.736244206369469*pi,1.240866963935217*pi) q[5];
U1q(0.541820401219375*pi,1.780211724121819*pi) q[6];
U1q(0.508036699590582*pi,1.74961649252739*pi) q[7];
U1q(0.585001007239609*pi,0.0137826334814558*pi) q[8];
U1q(0.662331979598059*pi,0.870388542737865*pi) q[9];
U1q(0.71427690737997*pi,1.468624227104157*pi) q[10];
U1q(0.425060740455884*pi,1.236964433219344*pi) q[11];
U1q(0.506444180394266*pi,0.865052400043981*pi) q[12];
U1q(0.283598056429209*pi,0.562936977178034*pi) q[13];
U1q(0.793475434531626*pi,1.359209712166471*pi) q[14];
U1q(0.353435787664257*pi,1.847329878465699*pi) q[15];
RZZ(0.0*pi) q[4],q[0];
RZZ(0.0*pi) q[2],q[1];
RZZ(0.0*pi) q[3],q[15];
RZZ(0.0*pi) q[5],q[8];
RZZ(0.0*pi) q[14],q[6];
RZZ(0.0*pi) q[10],q[7];
RZZ(0.0*pi) q[9],q[11];
RZZ(0.0*pi) q[13],q[12];
rz(0.780636112175594*pi) q[0];
rz(0.021951253466249*pi) q[1];
rz(3.786085673827728*pi) q[2];
rz(0.271986429888465*pi) q[3];
rz(0.712244443440566*pi) q[4];
rz(0.170299166657035*pi) q[5];
rz(1.14595892825532*pi) q[6];
rz(3.837607669490957*pi) q[7];
rz(1.41849193058681*pi) q[8];
rz(3.84458885853224*pi) q[9];
rz(3.186017962787693*pi) q[10];
rz(2.56294445332687*pi) q[11];
rz(0.217917466678702*pi) q[12];
rz(1.17899229963481*pi) q[13];
rz(2.1800445127887897*pi) q[14];
rz(0.367181821835907*pi) q[15];
U1q(0.501010077881563*pi,0.288058692225066*pi) q[0];
U1q(0.137499016724385*pi,0.400133688294165*pi) q[1];
U1q(0.103152317316322*pi,1.279888377874351*pi) q[2];
U1q(0.270322094069575*pi,1.836857282867568*pi) q[3];
U1q(0.496180112712549*pi,1.710933891334438*pi) q[4];
U1q(0.755053864072277*pi,0.745274105931272*pi) q[5];
U1q(0.324740725954127*pi,1.43535296768309*pi) q[6];
U1q(0.290764984448417*pi,0.367793171353479*pi) q[7];
U1q(0.975746521245889*pi,1.38758290573789*pi) q[8];
U1q(0.536229383934902*pi,0.332238466275979*pi) q[9];
U1q(0.621515839474076*pi,1.9863405100454914*pi) q[10];
U1q(0.531646650025752*pi,1.156597062445919*pi) q[11];
U1q(0.671427287439085*pi,0.131199331815419*pi) q[12];
U1q(0.382926898608938*pi,1.57074905480954*pi) q[13];
U1q(0.723770281433861*pi,1.819385400540968*pi) q[14];
U1q(0.571981965601822*pi,0.737092932226938*pi) q[15];
RZZ(0.0*pi) q[0],q[12];
RZZ(0.0*pi) q[3],q[1];
RZZ(0.0*pi) q[5],q[2];
RZZ(0.0*pi) q[4],q[10];
RZZ(0.0*pi) q[11],q[6];
RZZ(0.0*pi) q[13],q[7];
RZZ(0.0*pi) q[14],q[8];
RZZ(0.0*pi) q[9],q[15];
rz(0.657177581716364*pi) q[0];
rz(3.712406225151545*pi) q[1];
rz(0.115382655665045*pi) q[2];
rz(2.62063462442047*pi) q[3];
rz(1.39136599144995*pi) q[4];
rz(0.354664328621987*pi) q[5];
rz(3.975768153556551*pi) q[6];
rz(1.39271351909131*pi) q[7];
rz(1.44061137986059*pi) q[8];
rz(1.80157230730878*pi) q[9];
rz(0.878705784303981*pi) q[10];
rz(0.132876894483383*pi) q[11];
rz(3.03460855207699*pi) q[12];
rz(2.54952505480023*pi) q[13];
rz(3.65746756471551*pi) q[14];
rz(0.965526460647713*pi) q[15];
U1q(0.504179211775194*pi,0.633817647011997*pi) q[0];
U1q(0.316382901909978*pi,0.143397888689503*pi) q[1];
U1q(0.317114163973615*pi,1.421680046204453*pi) q[2];
U1q(0.922048374832388*pi,0.00344305219241764*pi) q[3];
U1q(0.147277943014496*pi,1.9141079063673394*pi) q[4];
U1q(0.252404551301506*pi,0.626126547458251*pi) q[5];
U1q(0.545128435736827*pi,0.162599401349004*pi) q[6];
U1q(0.442740764840632*pi,0.102588711892311*pi) q[7];
U1q(0.282708779652201*pi,0.0500055472939779*pi) q[8];
U1q(0.561804668380612*pi,0.880258525238044*pi) q[9];
U1q(0.313509767234272*pi,1.9506275023637587*pi) q[10];
U1q(0.348818119638963*pi,0.930289095230645*pi) q[11];
U1q(0.619911682326614*pi,1.656600479777313*pi) q[12];
U1q(0.662099092953096*pi,1.381165656979693*pi) q[13];
U1q(0.502739899933062*pi,0.523814436721679*pi) q[14];
U1q(0.448584315164568*pi,1.10629329295166*pi) q[15];
RZZ(0.0*pi) q[0],q[11];
RZZ(0.0*pi) q[9],q[1];
RZZ(0.0*pi) q[2],q[8];
RZZ(0.0*pi) q[3],q[5];
RZZ(0.0*pi) q[4],q[7];
RZZ(0.0*pi) q[13],q[6];
RZZ(0.0*pi) q[10],q[14];
RZZ(0.0*pi) q[15],q[12];
rz(2.92762428301894*pi) q[0];
rz(0.778704130640544*pi) q[1];
rz(0.0631231428974252*pi) q[2];
rz(2.48754320490358*pi) q[3];
rz(2.7807671234049*pi) q[4];
rz(0.573551336915118*pi) q[5];
rz(1.33419256996098*pi) q[6];
rz(0.893505497479901*pi) q[7];
rz(0.774452281352574*pi) q[8];
rz(3.470474993333641*pi) q[9];
rz(1.73027575492571*pi) q[10];
rz(0.39564026012251*pi) q[11];
rz(0.529278292944681*pi) q[12];
rz(3.598594591122023*pi) q[13];
rz(1.14453727702595*pi) q[14];
rz(1.0854684168259*pi) q[15];
U1q(0.658239386572528*pi,1.9729688628992412*pi) q[0];
U1q(0.826794337318634*pi,0.618935495720937*pi) q[1];
U1q(0.717171512738963*pi,0.519070440928244*pi) q[2];
U1q(0.626446281333557*pi,1.845529965371258*pi) q[3];
U1q(0.508355126803352*pi,1.9181656651670287*pi) q[4];
U1q(0.431595885917528*pi,1.4412848383963301*pi) q[5];
U1q(0.532501569818392*pi,0.396498385297992*pi) q[6];
U1q(0.653064795358156*pi,0.973867894840143*pi) q[7];
U1q(0.124339674559087*pi,1.400760144324018*pi) q[8];
U1q(0.506804680437744*pi,0.0122463166532178*pi) q[9];
U1q(0.628647474633406*pi,0.669524044493287*pi) q[10];
U1q(0.555878302822195*pi,0.605846082608755*pi) q[11];
U1q(0.203066248024849*pi,0.44529733791101*pi) q[12];
U1q(0.3303895345645*pi,0.186378404516008*pi) q[13];
U1q(0.276698419670144*pi,1.1044059927907*pi) q[14];
U1q(0.557474521043627*pi,1.33511794515163*pi) q[15];
RZZ(0.0*pi) q[13],q[0];
RZZ(0.0*pi) q[1],q[15];
RZZ(0.0*pi) q[2],q[7];
RZZ(0.0*pi) q[3],q[11];
RZZ(0.0*pi) q[4],q[9];
RZZ(0.0*pi) q[5],q[14];
RZZ(0.0*pi) q[10],q[6];
RZZ(0.0*pi) q[8],q[12];
rz(3.061658176577141*pi) q[0];
rz(1.43589648058301*pi) q[1];
rz(0.122102089196814*pi) q[2];
rz(3.740769580938304*pi) q[3];
rz(1.69390897326583*pi) q[4];
rz(0.695083526105869*pi) q[5];
rz(0.117596907426978*pi) q[6];
rz(3.548894429740573*pi) q[7];
rz(3.879325903449022*pi) q[8];
rz(3.725700033611805*pi) q[9];
rz(1.2721007840326*pi) q[10];
rz(1.29587820164048*pi) q[11];
rz(0.449402439030097*pi) q[12];
rz(2.35531814509321*pi) q[13];
rz(0.454620497176498*pi) q[14];
rz(1.1066876903432*pi) q[15];
U1q(0.603012048194443*pi,1.702387144263457*pi) q[0];
U1q(0.690383262249379*pi,1.18794047573115*pi) q[1];
U1q(0.461416661171763*pi,0.801497385857026*pi) q[2];
U1q(0.46541481521108*pi,0.00395695958702413*pi) q[3];
U1q(0.669047443761842*pi,1.3959343706896*pi) q[4];
U1q(0.378934707984133*pi,0.983208276978051*pi) q[5];
U1q(0.401330784171529*pi,0.777041840121915*pi) q[6];
U1q(0.381927498427378*pi,1.66220200455589*pi) q[7];
U1q(0.819165450828657*pi,0.35050956665344*pi) q[8];
U1q(0.0775337639076251*pi,0.0416610367755263*pi) q[9];
U1q(0.440986710910094*pi,1.1368025218998*pi) q[10];
U1q(0.615331125530325*pi,0.708470803310131*pi) q[11];
U1q(0.412158186038428*pi,1.07400168238095*pi) q[12];
U1q(0.504925267658887*pi,1.5342277667170219*pi) q[13];
U1q(0.314990099024922*pi,0.940158222059661*pi) q[14];
U1q(0.506940701975037*pi,0.915663004180646*pi) q[15];
RZZ(0.0*pi) q[0],q[6];
RZZ(0.0*pi) q[5],q[1];
RZZ(0.0*pi) q[9],q[2];
RZZ(0.0*pi) q[3],q[7];
RZZ(0.0*pi) q[4],q[14];
RZZ(0.0*pi) q[10],q[8];
RZZ(0.0*pi) q[11],q[12];
RZZ(0.0*pi) q[13],q[15];
rz(0.395426970228787*pi) q[0];
rz(0.909297982466342*pi) q[1];
rz(3.879609588297201*pi) q[2];
rz(1.25870870142809*pi) q[3];
rz(3.671041444150315*pi) q[4];
rz(0.281434562453344*pi) q[5];
rz(1.32359422346487*pi) q[6];
rz(1.32814658386592*pi) q[7];
rz(2.46312212964572*pi) q[8];
rz(3.736428837198087*pi) q[9];
rz(0.519593884549332*pi) q[10];
rz(3.556626035660087*pi) q[11];
rz(3.633223559172625*pi) q[12];
rz(0.13563839404228*pi) q[13];
rz(0.978399776564779*pi) q[14];
rz(3.735949764961492*pi) q[15];
U1q(0.353593529604847*pi,0.416506031108186*pi) q[0];
U1q(0.661708152162959*pi,0.269075364717613*pi) q[1];
U1q(0.465161053427287*pi,0.0164900477014795*pi) q[2];
U1q(0.555191163064459*pi,0.633574860129751*pi) q[3];
U1q(0.66378716599318*pi,1.9130868325452062*pi) q[4];
U1q(0.147748852400919*pi,1.1021532500486*pi) q[5];
U1q(0.397793729966271*pi,1.691859134310806*pi) q[6];
U1q(0.466904487052135*pi,0.0751579610941862*pi) q[7];
U1q(0.59424963667903*pi,1.319464908470559*pi) q[8];
U1q(0.340518068896173*pi,0.592872983100628*pi) q[9];
U1q(0.360078723914931*pi,1.924832962338234*pi) q[10];
U1q(0.201927507615121*pi,1.046598049073551*pi) q[11];
U1q(0.750764593445415*pi,0.160330019992088*pi) q[12];
U1q(0.208733099643358*pi,1.852324866313198*pi) q[13];
U1q(0.403607641720684*pi,1.30102387530531*pi) q[14];
U1q(0.452519764033848*pi,0.532742633859815*pi) q[15];
RZZ(0.0*pi) q[0],q[1];
RZZ(0.0*pi) q[14],q[2];
RZZ(0.0*pi) q[3],q[13];
RZZ(0.0*pi) q[4],q[5];
RZZ(0.0*pi) q[15],q[6];
RZZ(0.0*pi) q[7],q[8];
RZZ(0.0*pi) q[9],q[12];
RZZ(0.0*pi) q[10],q[11];
rz(0.52440478159237*pi) q[0];
rz(2.6550192869784697*pi) q[1];
rz(1.2566649391995*pi) q[2];
rz(0.526377666115273*pi) q[3];
rz(0.827591057458637*pi) q[4];
rz(3.697147277863272*pi) q[5];
rz(0.664003287923544*pi) q[6];
rz(1.09550815413452*pi) q[7];
rz(2.53586238766621*pi) q[8];
rz(0.170507522556178*pi) q[9];
rz(1.49017503994152*pi) q[10];
rz(1.28235293176896*pi) q[11];
rz(3.665254673836485*pi) q[12];
rz(1.46975055455711*pi) q[13];
rz(1.55287909836563*pi) q[14];
rz(3.746794427525849*pi) q[15];
U1q(0.757339829072982*pi,0.789337559211725*pi) q[0];
U1q(0.672685955500476*pi,0.0385803578946942*pi) q[1];
U1q(0.105995099496873*pi,1.720254011736745*pi) q[2];
U1q(0.152659866741201*pi,1.341576210485337*pi) q[3];
U1q(0.626751009848411*pi,0.900113173565909*pi) q[4];
U1q(0.459134734868019*pi,1.771992968937046*pi) q[5];
U1q(0.508749641212646*pi,0.0825875090571289*pi) q[6];
U1q(0.652938475572155*pi,0.776106990664446*pi) q[7];
U1q(0.769859153890525*pi,1.417889819239952*pi) q[8];
U1q(0.297622193749802*pi,0.623238111490261*pi) q[9];
U1q(0.287166591606852*pi,0.875964039454226*pi) q[10];
U1q(0.920523124163392*pi,0.562642968958961*pi) q[11];
U1q(0.40177698252765*pi,0.576980641419716*pi) q[12];
U1q(0.368483199647723*pi,0.0384033414060073*pi) q[13];
U1q(0.884657359252317*pi,0.958674944230303*pi) q[14];
U1q(0.277300323268065*pi,1.4345475841663*pi) q[15];
RZZ(0.0*pi) q[0],q[14];
RZZ(0.0*pi) q[1],q[12];
RZZ(0.0*pi) q[2],q[15];
RZZ(0.0*pi) q[4],q[3];
RZZ(0.0*pi) q[5],q[6];
RZZ(0.0*pi) q[9],q[7];
RZZ(0.0*pi) q[11],q[8];
RZZ(0.0*pi) q[13],q[10];
rz(0.89810490481414*pi) q[0];
rz(1.68284430696373*pi) q[1];
rz(0.97959928026302*pi) q[2];
rz(3.855784645400718*pi) q[3];
rz(0.549093540701401*pi) q[4];
rz(1.81643230698874*pi) q[5];
rz(3.163155842970951*pi) q[6];
rz(1.00191284556651*pi) q[7];
rz(2.0468268899475*pi) q[8];
rz(0.465115846672431*pi) q[9];
rz(3.9528616596665325*pi) q[10];
rz(2.45217247251896*pi) q[11];
rz(0.759282820642246*pi) q[12];
rz(0.627119127617713*pi) q[13];
rz(0.205848908268156*pi) q[14];
rz(1.1318486122533*pi) q[15];
U1q(0.303912726168702*pi,1.13597204957057*pi) q[0];
U1q(0.907775153962346*pi,1.40769951964072*pi) q[1];
U1q(0.291741753594859*pi,1.845266316524759*pi) q[2];
U1q(0.124125353209418*pi,1.539662825428373*pi) q[3];
U1q(0.438388297956674*pi,0.0658739969725222*pi) q[4];
U1q(0.56589893889189*pi,0.760483078148895*pi) q[5];
U1q(0.534568030466004*pi,1.5171659766870431*pi) q[6];
U1q(0.8565086101552*pi,0.361938843877512*pi) q[7];
U1q(0.583806541287453*pi,1.116918671307501*pi) q[8];
U1q(0.101401520887925*pi,1.17696162194029*pi) q[9];
U1q(0.159648117928646*pi,1.864840056325611*pi) q[10];
U1q(0.597026048524141*pi,1.54119393478626*pi) q[11];
U1q(0.589693305713756*pi,0.405947871177887*pi) q[12];
U1q(0.834917714427616*pi,0.661906353166004*pi) q[13];
U1q(0.393017212360546*pi,0.00809888019206007*pi) q[14];
U1q(0.188116825062675*pi,0.31940908770264*pi) q[15];
RZZ(0.0*pi) q[10],q[0];
RZZ(0.0*pi) q[1],q[7];
RZZ(0.0*pi) q[2],q[12];
RZZ(0.0*pi) q[9],q[3];
RZZ(0.0*pi) q[4],q[15];
RZZ(0.0*pi) q[13],q[5];
RZZ(0.0*pi) q[6],q[8];
RZZ(0.0*pi) q[14],q[11];
rz(1.7509423927995*pi) q[0];
rz(3.078892543772222*pi) q[1];
rz(1.53997055062659*pi) q[2];
rz(0.650383137548517*pi) q[3];
rz(3.637975369114982*pi) q[4];
rz(3.830091464421011*pi) q[5];
rz(3.859009339081142*pi) q[6];
rz(3.815727072594309*pi) q[7];
rz(1.94348813223889*pi) q[8];
rz(2.6524365993354797*pi) q[9];
rz(1.43087734645304*pi) q[10];
rz(3.712849966623977*pi) q[11];
rz(1.22881145786864*pi) q[12];
rz(0.533839736001268*pi) q[13];
rz(3.614894704892844*pi) q[14];
rz(3.707672949222035*pi) q[15];
U1q(0.802282616914173*pi,0.794447995874843*pi) q[0];
U1q(0.825565689907326*pi,1.59170424306514*pi) q[1];
U1q(0.740696993959519*pi,1.46763493997488*pi) q[2];
U1q(0.298899054680561*pi,1.687682960871939*pi) q[3];
U1q(0.683474265691186*pi,1.68635798605291*pi) q[4];
U1q(0.0722509875748995*pi,1.080508228056491*pi) q[5];
U1q(0.435537111109242*pi,0.418479151811578*pi) q[6];
U1q(0.685638423259877*pi,0.238983694956408*pi) q[7];
U1q(0.816973210879997*pi,1.21657357714119*pi) q[8];
U1q(0.80663054133114*pi,1.3294524275909438*pi) q[9];
U1q(0.464147172165185*pi,1.53771341430742*pi) q[10];
U1q(0.546643543472019*pi,0.203808061678213*pi) q[11];
U1q(0.342340923674321*pi,1.9941385699166112*pi) q[12];
U1q(0.344943947179504*pi,1.555942921073467*pi) q[13];
U1q(0.448168880166152*pi,0.124335498726838*pi) q[14];
U1q(0.11642463214179*pi,1.527662745952877*pi) q[15];
RZZ(0.0*pi) q[0],q[15];
RZZ(0.0*pi) q[13],q[1];
RZZ(0.0*pi) q[4],q[2];
RZZ(0.0*pi) q[3],q[8];
RZZ(0.0*pi) q[5],q[10];
RZZ(0.0*pi) q[9],q[6];
RZZ(0.0*pi) q[11],q[7];
RZZ(0.0*pi) q[14],q[12];
rz(1.32985291442507*pi) q[0];
rz(1.3880725559763*pi) q[1];
rz(1.58136366808952*pi) q[2];
rz(1.06388184284553*pi) q[3];
rz(3.9757092729416446*pi) q[4];
rz(2.75172785698126*pi) q[5];
rz(1.48368255764747*pi) q[6];
rz(1.46300683128532*pi) q[7];
rz(0.892643206473871*pi) q[8];
rz(3.911345487785472*pi) q[9];
rz(2.0513625752502698*pi) q[10];
rz(0.156162224559091*pi) q[11];
rz(1.19083448537407*pi) q[12];
rz(0.42744158707484*pi) q[13];
rz(0.812948896196551*pi) q[14];
rz(0.728544185133732*pi) q[15];
U1q(0.63736099371428*pi,0.957754064100819*pi) q[0];
U1q(0.825196838327223*pi,0.734806458796796*pi) q[1];
U1q(0.882544728637853*pi,0.984801327943086*pi) q[2];
U1q(0.396463176286905*pi,0.0628371705777189*pi) q[3];
U1q(0.405466643689009*pi,0.847668389088091*pi) q[4];
U1q(0.722134167042964*pi,0.0477468799311761*pi) q[5];
U1q(0.643932466920923*pi,1.01204307617609*pi) q[6];
U1q(0.741057154371897*pi,1.3106906214728*pi) q[7];
U1q(0.558034342482381*pi,1.01479184795079*pi) q[8];
U1q(0.374021214704956*pi,1.120830149970251*pi) q[9];
U1q(0.900104706810462*pi,1.283660261592534*pi) q[10];
U1q(0.244677245879453*pi,1.603015014832654*pi) q[11];
U1q(0.330746758894003*pi,0.22349493706035*pi) q[12];
U1q(0.721039925699644*pi,0.0910359990971621*pi) q[13];
U1q(0.584277364607058*pi,1.03239446114146*pi) q[14];
U1q(0.799771617168707*pi,0.22087314377906*pi) q[15];
RZZ(0.0*pi) q[0],q[7];
RZZ(0.0*pi) q[1],q[8];
RZZ(0.0*pi) q[3],q[2];
RZZ(0.0*pi) q[4],q[6];
RZZ(0.0*pi) q[5],q[12];
RZZ(0.0*pi) q[9],q[10];
RZZ(0.0*pi) q[13],q[11];
RZZ(0.0*pi) q[14],q[15];
rz(2.71900036265124*pi) q[0];
rz(1.47144473953033*pi) q[1];
rz(0.210870803160876*pi) q[2];
rz(0.738472586029601*pi) q[3];
rz(1.38467021047798*pi) q[4];
rz(3.790409841554851*pi) q[5];
rz(3.84131977821422*pi) q[6];
rz(0.266020181577963*pi) q[7];
rz(1.90014397540395*pi) q[8];
rz(3.640199160332269*pi) q[9];
rz(1.09030180339902*pi) q[10];
rz(0.660152020250161*pi) q[11];
rz(1.14559960411729*pi) q[12];
rz(1.26977969090501*pi) q[13];
rz(1.54362215564786*pi) q[14];
rz(3.6326188717644072*pi) q[15];
U1q(0.569513842492858*pi,1.642270829389128*pi) q[0];
U1q(0.417959401396845*pi,0.758780710146243*pi) q[1];
U1q(0.329502447637248*pi,1.953711776059344*pi) q[2];
U1q(0.569112886276108*pi,0.796961203167658*pi) q[3];
U1q(0.413037249658114*pi,1.909989585943422*pi) q[4];
U1q(0.267388355253342*pi,0.168503580696296*pi) q[5];
U1q(0.46252504317999*pi,0.818271401315586*pi) q[6];
U1q(0.247576275979068*pi,0.540475929705478*pi) q[7];
U1q(0.645154282907082*pi,1.63946602137893*pi) q[8];
U1q(0.385126353045978*pi,0.161133990346973*pi) q[9];
U1q(0.69112887071573*pi,0.592506482722113*pi) q[10];
U1q(0.509404180097454*pi,0.781648197747787*pi) q[11];
U1q(0.481164583906945*pi,1.31069379389419*pi) q[12];
U1q(0.535648761971863*pi,0.938712665857957*pi) q[13];
U1q(0.606833849420725*pi,1.2296947155157*pi) q[14];
U1q(0.392958550524299*pi,1.888582991893714*pi) q[15];
RZZ(0.0*pi) q[9],q[0];
RZZ(0.0*pi) q[14],q[1];
RZZ(0.0*pi) q[2],q[11];
RZZ(0.0*pi) q[3],q[10];
RZZ(0.0*pi) q[4],q[12];
RZZ(0.0*pi) q[5],q[15];
RZZ(0.0*pi) q[6],q[7];
RZZ(0.0*pi) q[13],q[8];
rz(0.517873269663409*pi) q[0];
rz(1.01802578744942*pi) q[1];
rz(0.763131385803288*pi) q[2];
rz(3.147372075314885*pi) q[3];
rz(0.465288382518316*pi) q[4];
rz(2.46947383880945*pi) q[5];
rz(0.809839880000767*pi) q[6];
rz(3.548754486649087*pi) q[7];
rz(0.720872360305463*pi) q[8];
rz(3.681839982420682*pi) q[9];
rz(2.5412091767915*pi) q[10];
rz(3.631643086626464*pi) q[11];
rz(0.89259077292928*pi) q[12];
rz(3.837764295627243*pi) q[13];
rz(0.184778396912481*pi) q[14];
rz(0.0469398834318757*pi) q[15];
U1q(0.0895990265730507*pi,0.610924550775074*pi) q[0];
U1q(0.548956140207278*pi,1.28013804156053*pi) q[1];
U1q(0.430630259815094*pi,1.20532426244169*pi) q[2];
U1q(0.634335233152523*pi,0.219541646768256*pi) q[3];
U1q(0.383263526963428*pi,1.9886128925520492*pi) q[4];
U1q(0.643774655038366*pi,1.713477206281935*pi) q[5];
U1q(0.482512625677131*pi,1.888454730952847*pi) q[6];
U1q(0.848296959733164*pi,0.0349266372069087*pi) q[7];
U1q(0.272777189396281*pi,0.407310925199106*pi) q[8];
U1q(0.157178787584734*pi,0.85075845198312*pi) q[9];
U1q(0.721003464791434*pi,1.436387627659193*pi) q[10];
U1q(0.146790186782226*pi,0.704894663958092*pi) q[11];
U1q(0.448488062712376*pi,0.623580153313939*pi) q[12];
U1q(0.408052453802887*pi,0.760947969203064*pi) q[13];
U1q(0.541187317228281*pi,0.134502623377186*pi) q[14];
U1q(0.556614316125565*pi,0.792871449020013*pi) q[15];
RZZ(0.0*pi) q[0],q[8];
RZZ(0.0*pi) q[4],q[1];
RZZ(0.0*pi) q[13],q[2];
RZZ(0.0*pi) q[3],q[6];
RZZ(0.0*pi) q[5],q[7];
RZZ(0.0*pi) q[9],q[14];
RZZ(0.0*pi) q[10],q[12];
RZZ(0.0*pi) q[11],q[15];
rz(0.800072540084635*pi) q[0];
rz(2.30275327519157*pi) q[1];
rz(3.506152530435103*pi) q[2];
rz(3.280831246088213*pi) q[3];
rz(1.34109631387337*pi) q[4];
rz(1.45639017832877*pi) q[5];
rz(0.817674286142984*pi) q[6];
rz(0.0213731131327807*pi) q[7];
rz(3.116646772685365*pi) q[8];
rz(0.729786648312051*pi) q[9];
rz(1.17226392948403*pi) q[10];
rz(1.33775677372748*pi) q[11];
rz(1.45515969564926*pi) q[12];
rz(2.15271381086994*pi) q[13];
rz(0.484336357095822*pi) q[14];
rz(1.24241364559217*pi) q[15];
U1q(0.303394694515252*pi,1.7803742291602491*pi) q[0];
U1q(0.670727299016217*pi,0.92098292017436*pi) q[1];
U1q(0.562377713978335*pi,1.831436057209799*pi) q[2];
U1q(0.679883857990937*pi,1.838551037380947*pi) q[3];
U1q(0.196536015499929*pi,1.51415602026373*pi) q[4];
U1q(0.415917754206791*pi,1.9724665267342512*pi) q[5];
U1q(0.226359704003545*pi,1.600042121036545*pi) q[6];
U1q(0.350539362495573*pi,0.6906431068561*pi) q[7];
U1q(0.581566286279251*pi,1.459380097460047*pi) q[8];
U1q(0.300571778632529*pi,1.28331667470347*pi) q[9];
U1q(0.754216867752451*pi,1.24843443040028*pi) q[10];
U1q(0.834585928710446*pi,0.761952428207851*pi) q[11];
U1q(0.885774236887877*pi,0.764793504836595*pi) q[12];
U1q(0.767772473446243*pi,0.88287457371766*pi) q[13];
U1q(0.257728208494288*pi,0.0916954814123308*pi) q[14];
U1q(0.515571467006011*pi,1.779129351004506*pi) q[15];
RZZ(0.0*pi) q[5],q[0];
RZZ(0.0*pi) q[10],q[1];
RZZ(0.0*pi) q[2],q[6];
RZZ(0.0*pi) q[3],q[14];
RZZ(0.0*pi) q[4],q[11];
RZZ(0.0*pi) q[7],q[12];
RZZ(0.0*pi) q[15],q[8];
RZZ(0.0*pi) q[9],q[13];
rz(3.81221541607124*pi) q[0];
rz(0.257845023110756*pi) q[1];
rz(0.538662744013106*pi) q[2];
rz(1.61246418319745*pi) q[3];
rz(3.3783170258085837*pi) q[4];
rz(3.31533577080896*pi) q[5];
rz(0.963880350949131*pi) q[6];
rz(3.507330548017121*pi) q[7];
rz(0.386741295808348*pi) q[8];
rz(2.01155806069541*pi) q[9];
rz(3.21506331922062*pi) q[10];
rz(1.28056371320572*pi) q[11];
rz(0.633037669136468*pi) q[12];
rz(3.73595141150825*pi) q[13];
rz(3.470307762012798*pi) q[14];
rz(1.37531445586368*pi) q[15];
U1q(3.343501336556956*pi,1.84080235326311*pi) q[0];
U1q(3.868984407962129*pi,1.469524426097193*pi) q[1];
U1q(3.344722449843004*pi,0.929862986387654*pi) q[2];
U1q(3.70437883556899*pi,1.89711635535444*pi) q[3];
U1q(3.743351410053258*pi,0.735183692393655*pi) q[4];
U1q(3.416406259571901*pi,1.52273463741969*pi) q[5];
U1q(3.829722631623824*pi,1.61386213867878*pi) q[6];
U1q(3.836978525049862*pi,0.0855473253919024*pi) q[7];
U1q(3.193704236499054*pi,0.754987818419231*pi) q[8];
U1q(3.514028338792242*pi,0.558401693406337*pi) q[9];
U1q(3.534651725408082*pi,1.562668672496411*pi) q[10];
U1q(3.136774383671418*pi,1.02376474975124*pi) q[11];
U1q(3.399067500233886*pi,0.752477046913216*pi) q[12];
U1q(3.501282396037559*pi,0.467479856775565*pi) q[13];
U1q(3.626256506051798*pi,1.32173400147054*pi) q[14];
U1q(3.456871349220852*pi,0.254214220101461*pi) q[15];
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