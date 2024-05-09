OPENQASM 2.0;
include "hqslib1.inc";

qreg q[24];
creg c[24];
U1q(0.275231550838879*pi,0.84513367746561*pi) q[0];
U1q(0.895931852838572*pi,1.3535536080551*pi) q[1];
U1q(1.43087627443576*pi,0.31274794974345815*pi) q[2];
U1q(0.586876153760084*pi,0.775804049660273*pi) q[3];
U1q(1.83514594301905*pi,0.3081783066587438*pi) q[4];
U1q(1.54109157950542*pi,0.7571829904532881*pi) q[5];
U1q(1.80102183338978*pi,0.21475649829548038*pi) q[6];
U1q(0.283302542329255*pi,0.74808980087179*pi) q[7];
U1q(0.439337603008929*pi,1.238571830371779*pi) q[8];
U1q(0.491507817926141*pi,1.6158654086711501*pi) q[9];
U1q(0.789594362110116*pi,0.734131760901968*pi) q[10];
U1q(3.649595141786819*pi,0.7704854743615716*pi) q[11];
U1q(1.67980265741937*pi,1.203839716638814*pi) q[12];
U1q(0.224596566822312*pi,1.9829234674936629*pi) q[13];
U1q(1.49182018627989*pi,1.49421939330183*pi) q[14];
U1q(1.22459591513328*pi,1.3671603218737203*pi) q[15];
U1q(0.0492052831215104*pi,1.447442524990523*pi) q[16];
U1q(0.314161913314502*pi,1.17703009294375*pi) q[17];
U1q(0.504741571916158*pi,1.9484084742054524*pi) q[18];
U1q(1.79236542174526*pi,1.2914527584333342*pi) q[19];
U1q(3.714999933450978*pi,1.0658512843801855*pi) q[20];
U1q(0.472485081339763*pi,1.4668177520809*pi) q[21];
U1q(1.57290182641224*pi,1.9574221378108023*pi) q[22];
U1q(1.44232295733182*pi,0.30653950786442974*pi) q[23];
RZZ(0.5*pi) q[0],q[8];
RZZ(0.5*pi) q[15],q[1];
RZZ(0.5*pi) q[2],q[12];
RZZ(0.5*pi) q[3],q[6];
RZZ(0.5*pi) q[16],q[4];
RZZ(0.5*pi) q[5],q[21];
RZZ(0.5*pi) q[7],q[18];
RZZ(0.5*pi) q[23],q[9];
RZZ(0.5*pi) q[11],q[10];
RZZ(0.5*pi) q[20],q[13];
RZZ(0.5*pi) q[14],q[19];
RZZ(0.5*pi) q[17],q[22];
U1q(0.893267680982527*pi,0.5993916880856802*pi) q[0];
U1q(0.417247883833291*pi,1.574463007777868*pi) q[1];
U1q(0.599841340281683*pi,0.5881304757776882*pi) q[2];
U1q(0.502183883730475*pi,1.14415565208941*pi) q[3];
U1q(0.557775287589128*pi,1.8973777600750834*pi) q[4];
U1q(0.607644036179723*pi,1.4744432499331082*pi) q[5];
U1q(0.228892412549609*pi,1.43515086144056*pi) q[6];
U1q(0.348647536482061*pi,1.19752336304781*pi) q[7];
U1q(0.178501714792378*pi,1.44008984422547*pi) q[8];
U1q(0.239482018492731*pi,1.29962347735289*pi) q[9];
U1q(0.8108520902264*pi,0.202979806847897*pi) q[10];
U1q(0.364014273067972*pi,1.6042951492972115*pi) q[11];
U1q(0.184130942829291*pi,1.3671382555068532*pi) q[12];
U1q(0.518487622707579*pi,0.84247292960002*pi) q[13];
U1q(0.198493962870704*pi,1.5313573060464094*pi) q[14];
U1q(0.336825738313018*pi,0.6248337607403904*pi) q[15];
U1q(0.61172058153655*pi,1.49635658348627*pi) q[16];
U1q(0.581047660941979*pi,1.91373905207127*pi) q[17];
U1q(0.533028442862187*pi,0.6439015304415898*pi) q[18];
U1q(0.358181682955327*pi,0.003919409793714301*pi) q[19];
U1q(0.49385689343036*pi,0.19506283473597552*pi) q[20];
U1q(0.418317900657646*pi,0.7391160742932001*pi) q[21];
U1q(0.587500266123367*pi,0.9024171031262225*pi) q[22];
U1q(0.355292414085127*pi,1.8159791163536498*pi) q[23];
RZZ(0.5*pi) q[0],q[9];
RZZ(0.5*pi) q[2],q[1];
RZZ(0.5*pi) q[3],q[14];
RZZ(0.5*pi) q[4],q[8];
RZZ(0.5*pi) q[19],q[5];
RZZ(0.5*pi) q[16],q[6];
RZZ(0.5*pi) q[7],q[15];
RZZ(0.5*pi) q[10],q[22];
RZZ(0.5*pi) q[13],q[11];
RZZ(0.5*pi) q[12],q[17];
RZZ(0.5*pi) q[18],q[21];
RZZ(0.5*pi) q[20],q[23];
U1q(0.774805235098871*pi,1.57874431822908*pi) q[0];
U1q(0.214365246493797*pi,1.56622964480073*pi) q[1];
U1q(0.573541135727495*pi,1.1171888222716282*pi) q[2];
U1q(0.507645526266006*pi,1.498588815488982*pi) q[3];
U1q(0.358874443945321*pi,0.3950109335626042*pi) q[4];
U1q(0.804078700557076*pi,1.5964970741983482*pi) q[5];
U1q(0.234636374693273*pi,1.6537008088117702*pi) q[6];
U1q(0.403554395870646*pi,0.7603360018017398*pi) q[7];
U1q(0.377121015080823*pi,1.23273913848071*pi) q[8];
U1q(0.343655598432728*pi,1.57937736546919*pi) q[9];
U1q(0.349872194245127*pi,0.5463432430218*pi) q[10];
U1q(0.462497633615861*pi,0.6405441268587819*pi) q[11];
U1q(0.528800485105898*pi,1.8741196518921641*pi) q[12];
U1q(0.322940764671134*pi,1.0699343359600002*pi) q[13];
U1q(0.82870478371097*pi,1.1068605382384495*pi) q[14];
U1q(0.351169073963413*pi,0.08604520871634946*pi) q[15];
U1q(0.456372921846001*pi,0.8229053837746001*pi) q[16];
U1q(0.902786751047721*pi,1.1323844570366903*pi) q[17];
U1q(0.591390548410884*pi,1.1526886147419502*pi) q[18];
U1q(0.500362322978702*pi,1.9572220831065046*pi) q[19];
U1q(0.687942535623306*pi,1.2310825209896352*pi) q[20];
U1q(0.273555176552304*pi,0.41122445486483006*pi) q[21];
U1q(0.488205352601846*pi,1.0572296054009023*pi) q[22];
U1q(0.315901756391644*pi,1.23622006316511*pi) q[23];
RZZ(0.5*pi) q[0],q[2];
RZZ(0.5*pi) q[17],q[1];
RZZ(0.5*pi) q[3],q[13];
RZZ(0.5*pi) q[4],q[5];
RZZ(0.5*pi) q[9],q[6];
RZZ(0.5*pi) q[7],q[20];
RZZ(0.5*pi) q[8],q[22];
RZZ(0.5*pi) q[10],q[12];
RZZ(0.5*pi) q[15],q[11];
RZZ(0.5*pi) q[14],q[18];
RZZ(0.5*pi) q[16],q[21];
RZZ(0.5*pi) q[23],q[19];
U1q(0.584001056793749*pi,0.1383461192325104*pi) q[0];
U1q(0.426783308251275*pi,1.9798801581981103*pi) q[1];
U1q(0.411485133461174*pi,1.4259594395714483*pi) q[2];
U1q(0.528868832267952*pi,1.5892211640492602*pi) q[3];
U1q(0.0981435924598098*pi,1.2630817936682233*pi) q[4];
U1q(0.928263345296154*pi,1.7192014330418672*pi) q[5];
U1q(0.847808502768191*pi,0.5041301076269313*pi) q[6];
U1q(0.456497500673434*pi,1.2880144933503406*pi) q[7];
U1q(0.499521579400046*pi,1.5687132406214497*pi) q[8];
U1q(0.867863846261821*pi,1.5274412578590102*pi) q[9];
U1q(0.698109450377107*pi,1.84936424727641*pi) q[10];
U1q(0.41554962030015*pi,1.5929351766403315*pi) q[11];
U1q(0.7502608622492*pi,1.5940946175782837*pi) q[12];
U1q(0.308615281232518*pi,0.48802880977594043*pi) q[13];
U1q(0.984486852243873*pi,1.0880748599106393*pi) q[14];
U1q(0.368246511404904*pi,1.0190701127393211*pi) q[15];
U1q(0.210457817508414*pi,0.40441658205446984*pi) q[16];
U1q(0.157132935187566*pi,0.11529213910223035*pi) q[17];
U1q(0.696298255330449*pi,0.8708156342351394*pi) q[18];
U1q(0.218707596482896*pi,1.3978524255303935*pi) q[19];
U1q(0.703211525914528*pi,0.8670124042439653*pi) q[20];
U1q(0.78779767607287*pi,0.9262215061711903*pi) q[21];
U1q(0.218766705288624*pi,0.7505047185507925*pi) q[22];
U1q(0.58918919786859*pi,0.8101479083238994*pi) q[23];
RZZ(0.5*pi) q[0],q[5];
RZZ(0.5*pi) q[7],q[1];
RZZ(0.5*pi) q[2],q[17];
RZZ(0.5*pi) q[3],q[12];
RZZ(0.5*pi) q[4],q[15];
RZZ(0.5*pi) q[6],q[22];
RZZ(0.5*pi) q[8],q[18];
RZZ(0.5*pi) q[9],q[21];
RZZ(0.5*pi) q[19],q[10];
RZZ(0.5*pi) q[20],q[11];
RZZ(0.5*pi) q[23],q[13];
RZZ(0.5*pi) q[16],q[14];
U1q(0.345788628886954*pi,0.9408877105862299*pi) q[0];
U1q(0.3875000490685*pi,1.5490223669212106*pi) q[1];
U1q(0.623881460327913*pi,1.2135890180120583*pi) q[2];
U1q(0.529817917055023*pi,1.5713094406219401*pi) q[3];
U1q(0.1632815125192*pi,1.9060209579083853*pi) q[4];
U1q(0.129160071704128*pi,0.9269519870737888*pi) q[5];
U1q(0.32994074996388*pi,0.06095778375528127*pi) q[6];
U1q(0.321121082467124*pi,1.8771182077908009*pi) q[7];
U1q(0.840087630361685*pi,1.6280133437533095*pi) q[8];
U1q(0.779790211153683*pi,0.6067068228327201*pi) q[9];
U1q(0.291566444016259*pi,1.0464763436755602*pi) q[10];
U1q(0.655395733217442*pi,1.0601929027516315*pi) q[11];
U1q(0.470160335873436*pi,1.6482427178873742*pi) q[12];
U1q(0.681077467486513*pi,0.49995948646906996*pi) q[13];
U1q(0.599394939495098*pi,1.4453202250578396*pi) q[14];
U1q(0.543000966113643*pi,0.06859594699552041*pi) q[15];
U1q(0.306528159528594*pi,0.5053964958307002*pi) q[16];
U1q(0.576742571398993*pi,0.9991454900215295*pi) q[17];
U1q(0.190231217457403*pi,0.37260579940440053*pi) q[18];
U1q(0.479765385392009*pi,1.5876857451875335*pi) q[19];
U1q(0.640920095718257*pi,1.1007051741055953*pi) q[20];
U1q(0.25792147796201*pi,1.5241673789470909*pi) q[21];
U1q(0.301588051814419*pi,1.5719868343212013*pi) q[22];
U1q(0.939144885840764*pi,1.7677137428470306*pi) q[23];
RZZ(0.5*pi) q[0],q[6];
RZZ(0.5*pi) q[10],q[1];
RZZ(0.5*pi) q[2],q[5];
RZZ(0.5*pi) q[3],q[16];
RZZ(0.5*pi) q[4],q[19];
RZZ(0.5*pi) q[7],q[12];
RZZ(0.5*pi) q[8],q[17];
RZZ(0.5*pi) q[20],q[9];
RZZ(0.5*pi) q[11],q[22];
RZZ(0.5*pi) q[13],q[21];
RZZ(0.5*pi) q[23],q[14];
RZZ(0.5*pi) q[15],q[18];
U1q(0.648676593955642*pi,0.05764067647949922*pi) q[0];
U1q(0.22361390750506*pi,1.6072765586259994*pi) q[1];
U1q(0.611049976924653*pi,0.08299181768155961*pi) q[2];
U1q(0.840523364055279*pi,0.16568035093238986*pi) q[3];
U1q(0.106865436543694*pi,1.8988260289880436*pi) q[4];
U1q(0.76229764500189*pi,0.12200109001686776*pi) q[5];
U1q(0.42640252111567*pi,1.596843712364981*pi) q[6];
U1q(0.377390967426766*pi,0.9031673252695995*pi) q[7];
U1q(0.610760046472558*pi,0.7712125169713602*pi) q[8];
U1q(0.7313257725228*pi,1.2088594754893691*pi) q[9];
U1q(0.0799941331900969*pi,0.3456537250295*pi) q[10];
U1q(0.214435973958174*pi,1.5316492119383813*pi) q[11];
U1q(0.137014685143638*pi,0.7423016178881152*pi) q[12];
U1q(0.806208226906103*pi,0.11462154455369955*pi) q[13];
U1q(0.149910098763742*pi,0.9745611740570297*pi) q[14];
U1q(0.360309824448865*pi,0.9986077554342199*pi) q[15];
U1q(0.553751310249161*pi,0.5380939773956008*pi) q[16];
U1q(0.370079214322933*pi,0.007751378186300784*pi) q[17];
U1q(0.756661902155263*pi,0.48671287938899965*pi) q[18];
U1q(0.153046168556922*pi,1.9618034522193337*pi) q[19];
U1q(0.568378532394655*pi,1.1404198441057858*pi) q[20];
U1q(0.344891694970693*pi,1.8111957614842993*pi) q[21];
U1q(0.281392602263807*pi,1.8657570006966022*pi) q[22];
U1q(0.675545936693516*pi,1.6611454412253313*pi) q[23];
RZZ(0.5*pi) q[3],q[0];
RZZ(0.5*pi) q[1],q[21];
RZZ(0.5*pi) q[2],q[11];
RZZ(0.5*pi) q[20],q[4];
RZZ(0.5*pi) q[23],q[5];
RZZ(0.5*pi) q[10],q[6];
RZZ(0.5*pi) q[7],q[14];
RZZ(0.5*pi) q[13],q[8];
RZZ(0.5*pi) q[12],q[9];
RZZ(0.5*pi) q[15],q[17];
RZZ(0.5*pi) q[16],q[18];
RZZ(0.5*pi) q[19],q[22];
U1q(0.357220159125323*pi,1.9914369294134993*pi) q[0];
U1q(0.453476710915204*pi,1.0103459335142002*pi) q[1];
U1q(0.717559095356425*pi,1.4904796871279586*pi) q[2];
U1q(0.431760203764956*pi,0.7553456798826002*pi) q[3];
U1q(0.500552094449516*pi,0.1085380340636437*pi) q[4];
U1q(0.556628515178209*pi,1.402175691519588*pi) q[5];
U1q(0.422439444133202*pi,0.8444156945366785*pi) q[6];
U1q(0.338081109775867*pi,0.5605633674617998*pi) q[7];
U1q(0.230192268850563*pi,0.8442497386506993*pi) q[8];
U1q(0.453092788439191*pi,1.8630715397505*pi) q[9];
U1q(0.481847034662099*pi,0.3927471223351695*pi) q[10];
U1q(0.842309768263893*pi,1.8694897102089705*pi) q[11];
U1q(0.540016183216722*pi,0.19176054993491398*pi) q[12];
U1q(0.885939790843765*pi,1.5460445882156009*pi) q[13];
U1q(0.574493718141047*pi,1.6643334911059302*pi) q[14];
U1q(0.791197567160834*pi,0.3859124842881201*pi) q[15];
U1q(0.387984948762025*pi,1.4395784072609992*pi) q[16];
U1q(0.468047349305904*pi,0.5590060786886006*pi) q[17];
U1q(0.536166254136566*pi,0.9884818074345993*pi) q[18];
U1q(0.635070340914756*pi,1.632036357322935*pi) q[19];
U1q(0.0977825055065109*pi,0.10943956766828578*pi) q[20];
U1q(0.571991292130423*pi,1.1371996337050998*pi) q[21];
U1q(0.610692043499744*pi,1.9891276027571045*pi) q[22];
U1q(0.735172703595847*pi,1.5690324753151312*pi) q[23];
rz(3.5767227272762003*pi) q[0];
rz(3.7138346554962*pi) q[1];
rz(0.4709060708524433*pi) q[2];
rz(3.3461323123145004*pi) q[3];
rz(0.24217430510335625*pi) q[4];
rz(1.2710413308395125*pi) q[5];
rz(1.1643268821357218*pi) q[6];
rz(0.01438067377510066*pi) q[7];
rz(0.3621074615947002*pi) q[8];
rz(2.9320602812807*pi) q[9];
rz(1.4144854374821403*pi) q[10];
rz(1.7146728273181289*pi) q[11];
rz(2.229669611291385*pi) q[12];
rz(2.5045720776156006*pi) q[13];
rz(3.74702786673717*pi) q[14];
rz(1.7361012935251807*pi) q[15];
rz(3.1748675428662008*pi) q[16];
rz(2.788558782010499*pi) q[17];
rz(3.338739772983301*pi) q[18];
rz(3.483815131062567*pi) q[19];
rz(0.9947559078833166*pi) q[20];
rz(0.8381069961238996*pi) q[21];
rz(1.5654647142292966*pi) q[22];
rz(3.3404358786403687*pi) q[23];
U1q(0.357220159125323*pi,0.568159656689674*pi) q[0];
U1q(0.453476710915204*pi,1.724180589010453*pi) q[1];
U1q(0.717559095356425*pi,0.9613857579804199*pi) q[2];
U1q(0.431760203764956*pi,1.101477992197165*pi) q[3];
U1q(1.50055209444952*pi,1.350712339167086*pi) q[4];
U1q(0.556628515178209*pi,1.673217022359119*pi) q[5];
U1q(1.4224394441332*pi,1.008742576672427*pi) q[6];
U1q(0.338081109775867*pi,1.5749440412369151*pi) q[7];
U1q(0.230192268850563*pi,0.206357200245329*pi) q[8];
U1q(1.45309278843919*pi,1.795131821031183*pi) q[9];
U1q(0.481847034662099*pi,0.80723255981731*pi) q[10];
U1q(0.842309768263893*pi,0.584162537527157*pi) q[11];
U1q(0.540016183216722*pi,1.421430161226377*pi) q[12];
U1q(0.885939790843765*pi,1.0506166658312521*pi) q[13];
U1q(1.57449371814105*pi,0.411361357843051*pi) q[14];
U1q(3.791197567160835*pi,1.12201377781334*pi) q[15];
U1q(0.387984948762025*pi,1.614445950127202*pi) q[16];
U1q(0.468047349305904*pi,0.347564860699098*pi) q[17];
U1q(1.53616625413657*pi,1.327221580417962*pi) q[18];
U1q(1.63507034091476*pi,0.115851488385517*pi) q[19];
U1q(1.09778250550651*pi,0.104195475551613*pi) q[20];
U1q(1.57199129213042*pi,0.975306629829036*pi) q[21];
U1q(0.610692043499744*pi,0.554592316986405*pi) q[22];
U1q(1.73517270359585*pi,1.9094683539555628*pi) q[23];
RZZ(0.5*pi) q[3],q[0];
RZZ(0.5*pi) q[1],q[21];
RZZ(0.5*pi) q[2],q[11];
RZZ(0.5*pi) q[20],q[4];
RZZ(0.5*pi) q[23],q[5];
RZZ(0.5*pi) q[10],q[6];
RZZ(0.5*pi) q[7],q[14];
RZZ(0.5*pi) q[13],q[8];
RZZ(0.5*pi) q[12],q[9];
RZZ(0.5*pi) q[15],q[17];
RZZ(0.5*pi) q[16],q[18];
RZZ(0.5*pi) q[19],q[22];
U1q(0.648676593955642*pi,1.634363403755638*pi) q[0];
U1q(0.22361390750506*pi,0.321111214122269*pi) q[1];
U1q(0.611049976924653*pi,0.5538978885339798*pi) q[2];
U1q(0.840523364055279*pi,1.5118126632469302*pi) q[3];
U1q(3.893134563456306*pi,1.5604243442427506*pi) q[4];
U1q(0.76229764500189*pi,0.39304242085640007*pi) q[5];
U1q(3.57359747888433*pi,1.2563145588441227*pi) q[6];
U1q(1.37739096742677*pi,1.9175479990447002*pi) q[7];
U1q(0.610760046472558*pi,1.1333199785660248*pi) q[8];
U1q(3.2686742274772*pi,1.4493438852923084*pi) q[9];
U1q(0.0799941331900969*pi,0.76013916251164*pi) q[10];
U1q(0.214435973958174*pi,1.2463220392565502*pi) q[11];
U1q(0.137014685143638*pi,1.97197122917951*pi) q[12];
U1q(1.8062082269061*pi,0.61919362216931*pi) q[13];
U1q(3.850089901236259*pi,0.10113367489189096*pi) q[14];
U1q(3.639690175551135*pi,1.509318506667306*pi) q[15];
U1q(1.55375131024916*pi,0.71296152026185*pi) q[16];
U1q(1.37007921432293*pi,1.79631016019685*pi) q[17];
U1q(1.75666190215526*pi,1.8289905084636047*pi) q[18];
U1q(1.15304616855692*pi,1.7860843934891515*pi) q[19];
U1q(1.56837853239466*pi,0.0732151991141251*pi) q[20];
U1q(3.655108305029307*pi,1.301310502049891*pi) q[21];
U1q(3.281392602263807*pi,0.4312217149259201*pi) q[22];
U1q(3.324454063306484*pi,0.8173553880453763*pi) q[23];
RZZ(0.5*pi) q[0],q[6];
RZZ(0.5*pi) q[10],q[1];
RZZ(0.5*pi) q[2],q[5];
RZZ(0.5*pi) q[3],q[16];
RZZ(0.5*pi) q[4],q[19];
RZZ(0.5*pi) q[7],q[12];
RZZ(0.5*pi) q[8],q[17];
RZZ(0.5*pi) q[20],q[9];
RZZ(0.5*pi) q[11],q[22];
RZZ(0.5*pi) q[13],q[21];
RZZ(0.5*pi) q[23],q[14];
RZZ(0.5*pi) q[15],q[18];
U1q(0.345788628886954*pi,0.51761043786239*pi) q[0];
U1q(1.3875000490685*pi,1.2628570224174411*pi) q[1];
U1q(0.623881460327913*pi,0.6844950888644803*pi) q[2];
U1q(0.529817917055023*pi,0.9174417529364902*pi) q[3];
U1q(3.836718487480799*pi,1.5532294153223898*pi) q[4];
U1q(0.129160071704128*pi,0.19799331791331998*pi) q[5];
U1q(1.32994074996388*pi,1.7922004874538482*pi) q[6];
U1q(3.678878917532876*pi,1.9435971165234838*pi) q[7];
U1q(0.840087630361685*pi,1.9901208053479702*pi) q[8];
U1q(1.77979021115368*pi,1.0514965379489567*pi) q[9];
U1q(3.291566444016259*pi,0.46096178115770003*pi) q[10];
U1q(0.655395733217442*pi,0.7748657300698001*pi) q[11];
U1q(0.470160335873436*pi,0.8779123291788098*pi) q[12];
U1q(1.68107746748651*pi,1.2338556802539324*pi) q[13];
U1q(1.5993949394951*pi,1.630374623891119*pi) q[14];
U1q(1.54300096611364*pi,1.439330315105997*pi) q[15];
U1q(3.693471840471406*pi,1.745659001826767*pi) q[16];
U1q(3.423257428601006*pi,1.804916048361628*pi) q[17];
U1q(3.190231217457403*pi,1.7148834284790029*pi) q[18];
U1q(0.479765385392009*pi,1.4119666864574247*pi) q[19];
U1q(0.640920095718257*pi,0.033500529113922095*pi) q[20];
U1q(3.25792147796201*pi,1.5883388845870776*pi) q[21];
U1q(3.698411948185581*pi,1.7249918813013627*pi) q[22];
U1q(1.93914488584076*pi,0.7107870864237005*pi) q[23];
RZZ(0.5*pi) q[0],q[5];
RZZ(0.5*pi) q[7],q[1];
RZZ(0.5*pi) q[2],q[17];
RZZ(0.5*pi) q[3],q[12];
RZZ(0.5*pi) q[4],q[15];
RZZ(0.5*pi) q[6],q[22];
RZZ(0.5*pi) q[8],q[18];
RZZ(0.5*pi) q[9],q[21];
RZZ(0.5*pi) q[19],q[10];
RZZ(0.5*pi) q[20],q[11];
RZZ(0.5*pi) q[23],q[13];
RZZ(0.5*pi) q[16],q[14];
U1q(1.58400105679375*pi,0.7150688465086601*pi) q[0];
U1q(3.573216691748725*pi,0.8319992311405437*pi) q[1];
U1q(0.411485133461174*pi,0.8968655104238596*pi) q[2];
U1q(1.52886883226795*pi,1.9353534763638*pi) q[3];
U1q(1.09814359245981*pi,1.1961685795625545*pi) q[4];
U1q(1.92826334529615*pi,1.9902427638814002*pi) q[5];
U1q(1.84780850276819*pi,0.23537281132551824*pi) q[6];
U1q(3.543502499326565*pi,1.5327008309639636*pi) q[7];
U1q(1.49952157940005*pi,1.9308207022161201*pi) q[8];
U1q(0.867863846261821*pi,0.9722309729752459*pi) q[9];
U1q(3.301890549622893*pi,0.6580738775568467*pi) q[10];
U1q(0.41554962030015*pi,0.3076080039585001*pi) q[11];
U1q(0.7502608622492*pi,1.8237642288697202*pi) q[12];
U1q(1.30861528123252*pi,0.2219250035608029*pi) q[13];
U1q(0.984486852243873*pi,1.2731292587439178*pi) q[14];
U1q(1.3682465114049*pi,0.389804480849831*pi) q[15];
U1q(3.789542182491585*pi,1.846638915602997*pi) q[16];
U1q(3.157132935187566*pi,1.6887693992809245*pi) q[17];
U1q(1.69629825533045*pi,1.216673593648233*pi) q[18];
U1q(0.218707596482896*pi,1.2221333668002448*pi) q[19];
U1q(1.70321152591453*pi,0.7998077592522925*pi) q[20];
U1q(0.78779767607287*pi,1.9903930118111695*pi) q[21];
U1q(3.781233294711376*pi,0.5464739970717529*pi) q[22];
U1q(1.58918919786859*pi,0.7532212519005705*pi) q[23];
RZZ(0.5*pi) q[0],q[2];
RZZ(0.5*pi) q[17],q[1];
RZZ(0.5*pi) q[3],q[13];
RZZ(0.5*pi) q[4],q[5];
RZZ(0.5*pi) q[9],q[6];
RZZ(0.5*pi) q[7],q[20];
RZZ(0.5*pi) q[8],q[22];
RZZ(0.5*pi) q[10],q[12];
RZZ(0.5*pi) q[15],q[11];
RZZ(0.5*pi) q[14],q[18];
RZZ(0.5*pi) q[16],q[21];
RZZ(0.5*pi) q[23],q[19];
U1q(3.225194764901129*pi,0.2746706475120906*pi) q[0];
U1q(3.785634753506202*pi,1.2456497445379138*pi) q[1];
U1q(1.57354113572749*pi,0.5880948931240404*pi) q[2];
U1q(1.50764552626601*pi,1.0259858249240734*pi) q[3];
U1q(0.358874443945321*pi,0.3280977194569248*pi) q[4];
U1q(3.195921299442924*pi,1.1129471227249228*pi) q[5];
U1q(1.23463637469327*pi,1.0858021101406758*pi) q[6];
U1q(1.40355439587065*pi,0.060379322512550626*pi) q[7];
U1q(1.37712101508082*pi,1.2667948043568615*pi) q[8];
U1q(3.343655598432727*pi,1.0241670805854257*pi) q[9];
U1q(3.349872194245127*pi,0.9610948818114586*pi) q[10];
U1q(1.46249763361586*pi,1.355216954176951*pi) q[11];
U1q(0.528800485105898*pi,0.10378926318359039*pi) q[12];
U1q(1.32294076467113*pi,0.6400194773767387*pi) q[13];
U1q(0.82870478371097*pi,0.291914937071728*pi) q[14];
U1q(1.35116907396341*pi,1.322829384872803*pi) q[15];
U1q(1.456372921846*pi,0.4281501138828627*pi) q[16];
U1q(3.902786751047721*pi,0.7058617172153747*pi) q[17];
U1q(1.59139054841088*pi,0.4985465741550428*pi) q[18];
U1q(0.500362322978702*pi,1.781503024376354*pi) q[19];
U1q(1.68794253562331*pi,1.4357376425066173*pi) q[20];
U1q(1.2735551765523*pi,1.4753959605048126*pi) q[21];
U1q(3.511794647398154*pi,1.2397491102216414*pi) q[22];
U1q(3.3159017563916438*pi,0.3271490970593689*pi) q[23];
RZZ(0.5*pi) q[0],q[9];
RZZ(0.5*pi) q[2],q[1];
RZZ(0.5*pi) q[3],q[14];
RZZ(0.5*pi) q[4],q[8];
RZZ(0.5*pi) q[19],q[5];
RZZ(0.5*pi) q[16],q[6];
RZZ(0.5*pi) q[7],q[15];
RZZ(0.5*pi) q[10],q[22];
RZZ(0.5*pi) q[13],q[11];
RZZ(0.5*pi) q[12],q[17];
RZZ(0.5*pi) q[18],q[21];
RZZ(0.5*pi) q[20],q[23];
U1q(3.106732319017474*pi,0.2540232776555005*pi) q[0];
U1q(3.582752116166709*pi,0.23741638156078348*pi) q[1];
U1q(3.4001586597183158*pi,1.1171532396179815*pi) q[2];
U1q(0.502183883730475*pi,1.6715526615245029*pi) q[3];
U1q(0.557775287589128*pi,0.8304645459694049*pi) q[4];
U1q(1.60764403617972*pi,0.23500094699016216*pi) q[5];
U1q(1.22889241254961*pi,1.867252162769466*pi) q[6];
U1q(0.348647536482061*pi,1.4975666837586212*pi) q[7];
U1q(0.178501714792378*pi,0.4741455101016214*pi) q[8];
U1q(3.760517981507269*pi,1.303920968701732*pi) q[9];
U1q(1.8108520902264*pi,0.617731445637558*pi) q[10];
U1q(1.36401427306797*pi,0.39146593173851585*pi) q[11];
U1q(0.184130942829291*pi,1.5968078667982795*pi) q[12];
U1q(0.518487622707579*pi,0.41255807101675845*pi) q[13];
U1q(0.198493962870704*pi,1.7164117048796883*pi) q[14];
U1q(1.33682573831302*pi,0.8616179368968429*pi) q[15];
U1q(1.61172058153655*pi,0.10160131359453306*pi) q[16];
U1q(3.4189523390580208*pi,0.9245071221807946*pi) q[17];
U1q(1.53302844286219*pi,0.0073336584554120066*pi) q[18];
U1q(0.358181682955327*pi,1.8282003510635638*pi) q[19];
U1q(0.49385689343036*pi,1.399717956252947*pi) q[20];
U1q(3.418317900657646*pi,0.14750434107644173*pi) q[21];
U1q(3.4124997338766327*pi,1.3945616124963216*pi) q[22];
U1q(1.35529241408513*pi,1.9069081502479097*pi) q[23];
RZZ(0.5*pi) q[0],q[8];
RZZ(0.5*pi) q[15],q[1];
RZZ(0.5*pi) q[2],q[12];
RZZ(0.5*pi) q[3],q[6];
RZZ(0.5*pi) q[16],q[4];
RZZ(0.5*pi) q[5],q[21];
RZZ(0.5*pi) q[7],q[18];
RZZ(0.5*pi) q[23],q[9];
RZZ(0.5*pi) q[11],q[10];
RZZ(0.5*pi) q[20],q[13];
RZZ(0.5*pi) q[14],q[19];
RZZ(0.5*pi) q[17],q[22];
U1q(3.275231550838879*pi,1.008281288275569*pi) q[0];
U1q(1.89593185283857*pi,0.45832578128355017*pi) q[1];
U1q(1.43087627443576*pi,0.39253576565221593*pi) q[2];
U1q(0.586876153760084*pi,1.3032010590953735*pi) q[3];
U1q(0.835145943019051*pi,1.241265092553065*pi) q[4];
U1q(0.541091579505422*pi,0.5177406875103436*pi) q[5];
U1q(1.80102183338978*pi,1.0876465259145434*pi) q[6];
U1q(0.283302542329255*pi,0.048133121582601746*pi) q[7];
U1q(0.439337603008929*pi,0.2726274962479316*pi) q[8];
U1q(1.49150781792614*pi,0.9876790373834794*pi) q[9];
U1q(1.78959436211012*pi,0.08657949158348788*pi) q[10];
U1q(0.649595141786819*pi,1.557656256802873*pi) q[11];
U1q(0.679802657419374*pi,1.4335093279301994*pi) q[12];
U1q(0.224596566822312*pi,0.5530086089103987*pi) q[13];
U1q(0.491820186279891*pi,0.6792737921351081*pi) q[14];
U1q(1.22459591513328*pi,1.1192913757635143*pi) q[15];
U1q(1.04920528312151*pi,0.1505153720902801*pi) q[16];
U1q(1.3141619133145*pi,1.6612160813083126*pi) q[17];
U1q(0.504741571916158*pi,0.3118406022192719*pi) q[18];
U1q(0.792365421745255*pi,0.11573369970318392*pi) q[19];
U1q(0.714999933450978*pi,1.2705064058971676*pi) q[20];
U1q(0.472485081339763*pi,1.8752060188641417*pi) q[21];
U1q(1.57290182641224*pi,0.33955657781173976*pi) q[22];
U1q(1.44232295733182*pi,0.4163477587371247*pi) q[23];
rz(0.991718711724431*pi) q[0];
rz(3.54167421871645*pi) q[1];
rz(1.607464234347784*pi) q[2];
rz(2.6967989409046265*pi) q[3];
rz(2.758734907446935*pi) q[4];
rz(3.4822593124896564*pi) q[5];
rz(2.9123534740854566*pi) q[6];
rz(1.9518668784173983*pi) q[7];
rz(3.7273725037520684*pi) q[8];
rz(3.0123209626165206*pi) q[9];
rz(1.9134205084165121*pi) q[10];
rz(0.4423437431971272*pi) q[11];
rz(2.5664906720698006*pi) q[12];
rz(3.4469913910896013*pi) q[13];
rz(3.320726207864892*pi) q[14];
rz(0.8807086242364857*pi) q[15];
rz(1.84948462790972*pi) q[16];
rz(2.3387839186916874*pi) q[17];
rz(1.688159397780728*pi) q[18];
rz(1.884266300296816*pi) q[19];
rz(0.7294935941028324*pi) q[20];
rz(2.1247939811358583*pi) q[21];
rz(1.6604434221882602*pi) q[22];
rz(1.5836522412628753*pi) q[23];
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