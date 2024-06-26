OPENQASM 2.0;
include "hqslib1.inc";

qreg q[56];
creg c[56];
U1q(0.705063566092912*pi,0.0977147007089494*pi) q[0];
U1q(0.477449330850189*pi,0.36088548088493*pi) q[1];
U1q(0.858839851933741*pi,1.708739548468683*pi) q[2];
U1q(0.894325579604509*pi,0.477704314593292*pi) q[3];
U1q(0.536793041499245*pi,1.9600184292685576*pi) q[4];
U1q(0.539435688026508*pi,0.388115932917449*pi) q[5];
U1q(0.346994591201229*pi,0.248472647137682*pi) q[6];
U1q(0.432640078217745*pi,0.145568872689305*pi) q[7];
U1q(0.356261813521782*pi,0.174569555073106*pi) q[8];
U1q(0.758306330202415*pi,0.558534139905077*pi) q[9];
U1q(0.629106687491275*pi,0.481875250224592*pi) q[10];
U1q(0.44792938097977*pi,0.9191297338942599*pi) q[11];
U1q(0.585794932416971*pi,0.0671078791928882*pi) q[12];
U1q(0.274456346745892*pi,1.12126830052486*pi) q[13];
U1q(0.113829440819286*pi,1.0139666684855602*pi) q[14];
U1q(0.835068245955134*pi,1.879104188490951*pi) q[15];
U1q(0.196584088200428*pi,1.044476700523244*pi) q[16];
U1q(0.633107508001746*pi,0.275098349765684*pi) q[17];
U1q(0.293912251478812*pi,0.97565000838856*pi) q[18];
U1q(0.731053646352641*pi,0.354912644733382*pi) q[19];
U1q(0.312889788570365*pi,1.531001035353075*pi) q[20];
U1q(0.39878801750623*pi,1.491901407929015*pi) q[21];
U1q(0.420573214019752*pi,0.98311472314129*pi) q[22];
U1q(0.451570208121392*pi,0.530364762332736*pi) q[23];
U1q(0.918900509480545*pi,1.82966688002841*pi) q[24];
U1q(0.405280673149233*pi,0.8886623881300499*pi) q[25];
U1q(0.564011269576358*pi,0.522747997984185*pi) q[26];
U1q(0.581104584878211*pi,0.538090572083656*pi) q[27];
U1q(0.507480983772289*pi,0.152958048619909*pi) q[28];
U1q(0.603992382126125*pi,0.701731599269843*pi) q[29];
U1q(0.175281307130145*pi,1.108079418706783*pi) q[30];
U1q(0.282055381743279*pi,0.124611254650117*pi) q[31];
U1q(0.543907925875439*pi,1.854684190125081*pi) q[32];
U1q(0.438235797617415*pi,0.784755555712131*pi) q[33];
U1q(0.591916884395626*pi,1.327362002175547*pi) q[34];
U1q(0.934171800845217*pi,0.0629586781544077*pi) q[35];
U1q(0.193605617341557*pi,0.602630435945859*pi) q[36];
U1q(0.815448817451844*pi,1.40845392599003*pi) q[37];
U1q(0.643285469551915*pi,0.699447051884264*pi) q[38];
U1q(0.373332873170185*pi,1.372529412083187*pi) q[39];
U1q(0.519576071458169*pi,1.36711916873755*pi) q[40];
U1q(0.194501432630479*pi,1.757392259337924*pi) q[41];
U1q(0.637078395010058*pi,0.979919676498338*pi) q[42];
U1q(0.475890315712781*pi,0.402694728580567*pi) q[43];
U1q(0.805871046012553*pi,0.978742171074327*pi) q[44];
U1q(0.538583181770452*pi,0.7617408583973799*pi) q[45];
U1q(0.64232695784325*pi,1.564270636017297*pi) q[46];
U1q(0.478780451877939*pi,1.4148947842853121*pi) q[47];
U1q(0.468080414304319*pi,0.595891003619247*pi) q[48];
U1q(0.318990003850591*pi,0.333102297774188*pi) q[49];
U1q(0.906667178467859*pi,0.807206240681953*pi) q[50];
U1q(0.966683085311468*pi,0.118687871343249*pi) q[51];
U1q(0.476941862240359*pi,1.6591365502659121*pi) q[52];
U1q(0.304299933115556*pi,1.897407196118245*pi) q[53];
U1q(0.34087618604124*pi,1.396665787552732*pi) q[54];
U1q(0.403451757774525*pi,0.77826962681078*pi) q[55];
RZZ(0.5*pi) q[34],q[0];
RZZ(0.5*pi) q[37],q[1];
RZZ(0.5*pi) q[2],q[12];
RZZ(0.5*pi) q[46],q[3];
RZZ(0.5*pi) q[4],q[44];
RZZ(0.5*pi) q[5],q[35];
RZZ(0.5*pi) q[40],q[6];
RZZ(0.5*pi) q[7],q[24];
RZZ(0.5*pi) q[8],q[18];
RZZ(0.5*pi) q[9],q[54];
RZZ(0.5*pi) q[14],q[10];
RZZ(0.5*pi) q[15],q[11];
RZZ(0.5*pi) q[13],q[49];
RZZ(0.5*pi) q[39],q[16];
RZZ(0.5*pi) q[17],q[48];
RZZ(0.5*pi) q[42],q[19];
RZZ(0.5*pi) q[20],q[22];
RZZ(0.5*pi) q[21],q[52];
RZZ(0.5*pi) q[51],q[23];
RZZ(0.5*pi) q[25],q[53];
RZZ(0.5*pi) q[26],q[33];
RZZ(0.5*pi) q[27],q[32];
RZZ(0.5*pi) q[28],q[43];
RZZ(0.5*pi) q[50],q[29];
RZZ(0.5*pi) q[30],q[31];
RZZ(0.5*pi) q[36],q[47];
RZZ(0.5*pi) q[38],q[45];
RZZ(0.5*pi) q[55],q[41];
U1q(0.753356826376403*pi,1.7504082255701698*pi) q[0];
U1q(0.523292354183012*pi,1.5997255792244198*pi) q[1];
U1q(0.269512507873488*pi,0.9332100496866698*pi) q[2];
U1q(0.619241565490503*pi,1.144948043042083*pi) q[3];
U1q(0.513544192577527*pi,0.7013476573381101*pi) q[4];
U1q(0.223035975290919*pi,1.230983082180644*pi) q[5];
U1q(0.512840999620544*pi,1.8809382656483602*pi) q[6];
U1q(0.650333017535375*pi,0.38818470031365004*pi) q[7];
U1q(0.465040699492868*pi,1.8286028826485898*pi) q[8];
U1q(0.59418588034049*pi,0.28048575692546995*pi) q[9];
U1q(0.28619950431354*pi,0.70000693615423*pi) q[10];
U1q(0.834924592275032*pi,0.41948048759698997*pi) q[11];
U1q(0.422507946713697*pi,0.36726499424913994*pi) q[12];
U1q(0.306051597225887*pi,1.88123257985443*pi) q[13];
U1q(0.624280771263054*pi,0.10111869526654993*pi) q[14];
U1q(0.698293143777452*pi,1.1714788358468402*pi) q[15];
U1q(0.317763638343189*pi,1.08374269182905*pi) q[16];
U1q(0.673066566369287*pi,1.21059300943132*pi) q[17];
U1q(0.656282963556494*pi,0.10247782319730003*pi) q[18];
U1q(0.559325487573415*pi,1.76594845810261*pi) q[19];
U1q(0.2933323439036*pi,1.1670130280694302*pi) q[20];
U1q(0.345218910722662*pi,1.23829972154598*pi) q[21];
U1q(0.296677143140028*pi,1.44276323794056*pi) q[22];
U1q(0.831811089163386*pi,1.41877059161386*pi) q[23];
U1q(0.305248921258938*pi,0.6387634882992601*pi) q[24];
U1q(0.645531020932014*pi,1.58103953389266*pi) q[25];
U1q(0.844778879302432*pi,0.1186940313484599*pi) q[26];
U1q(0.705631299840597*pi,1.663051262908255*pi) q[27];
U1q(0.0562688802714714*pi,1.41701869192686*pi) q[28];
U1q(0.473150392320337*pi,1.782844861601546*pi) q[29];
U1q(0.788243814442478*pi,0.5465149674338501*pi) q[30];
U1q(0.50818564981819*pi,0.32666286210209994*pi) q[31];
U1q(0.230707769467159*pi,0.22183369890057003*pi) q[32];
U1q(0.370525735483703*pi,1.0668911843646*pi) q[33];
U1q(0.519578666100861*pi,1.8887048164908098*pi) q[34];
U1q(0.309260496991488*pi,0.5358037694491098*pi) q[35];
U1q(0.335620023834944*pi,1.84999409340916*pi) q[36];
U1q(0.735483286323011*pi,0.114819508146482*pi) q[37];
U1q(0.0476745297476195*pi,1.3641634113530299*pi) q[38];
U1q(0.468115441627144*pi,1.72870617277293*pi) q[39];
U1q(0.493978652368862*pi,1.9109580845950893*pi) q[40];
U1q(0.725757534225747*pi,1.426888669205021*pi) q[41];
U1q(0.180064665866445*pi,0.3550288589694499*pi) q[42];
U1q(0.769998056360727*pi,0.19412781681004998*pi) q[43];
U1q(0.0887347248355943*pi,0.96990953628342*pi) q[44];
U1q(0.290038266042146*pi,1.4354735676951398*pi) q[45];
U1q(0.407896745463705*pi,1.0917631279487203*pi) q[46];
U1q(0.814683546062815*pi,1.49303520783993*pi) q[47];
U1q(0.553297951830331*pi,0.07996998204576*pi) q[48];
U1q(0.225125185048584*pi,0.41996806774359996*pi) q[49];
U1q(0.702929877435375*pi,1.097748292713077*pi) q[50];
U1q(0.511022043444697*pi,0.9255188664034999*pi) q[51];
U1q(0.542610363664775*pi,0.8636135058786398*pi) q[52];
U1q(0.32132855593811*pi,1.42410259521822*pi) q[53];
U1q(0.519149495764849*pi,1.1306064542537149*pi) q[54];
U1q(0.120913442706485*pi,1.7133372816288501*pi) q[55];
RZZ(0.5*pi) q[42],q[0];
RZZ(0.5*pi) q[9],q[1];
RZZ(0.5*pi) q[2],q[40];
RZZ(0.5*pi) q[52],q[3];
RZZ(0.5*pi) q[4],q[48];
RZZ(0.5*pi) q[12],q[5];
RZZ(0.5*pi) q[55],q[6];
RZZ(0.5*pi) q[7],q[17];
RZZ(0.5*pi) q[8],q[34];
RZZ(0.5*pi) q[25],q[10];
RZZ(0.5*pi) q[37],q[11];
RZZ(0.5*pi) q[39],q[13];
RZZ(0.5*pi) q[14],q[35];
RZZ(0.5*pi) q[15],q[20];
RZZ(0.5*pi) q[16],q[45];
RZZ(0.5*pi) q[30],q[18];
RZZ(0.5*pi) q[33],q[19];
RZZ(0.5*pi) q[21],q[41];
RZZ(0.5*pi) q[22],q[43];
RZZ(0.5*pi) q[28],q[23];
RZZ(0.5*pi) q[24],q[51];
RZZ(0.5*pi) q[26],q[47];
RZZ(0.5*pi) q[27],q[50];
RZZ(0.5*pi) q[29],q[32];
RZZ(0.5*pi) q[31],q[38];
RZZ(0.5*pi) q[36],q[49];
RZZ(0.5*pi) q[46],q[44];
RZZ(0.5*pi) q[53],q[54];
U1q(0.773843544436505*pi,1.0018831865055597*pi) q[0];
U1q(0.325741465236698*pi,1.0757474999758*pi) q[1];
U1q(0.434796345135701*pi,1.5494468088403002*pi) q[2];
U1q(0.185940904455535*pi,1.1908401645134101*pi) q[3];
U1q(0.244943320529596*pi,0.4632877738088803*pi) q[4];
U1q(0.501608970798327*pi,0.6754713457183099*pi) q[5];
U1q(0.639452953257686*pi,0.6110968446396101*pi) q[6];
U1q(0.256020880871501*pi,0.5595242658898498*pi) q[7];
U1q(0.698442200826145*pi,1.1858296245686804*pi) q[8];
U1q(0.167560303526431*pi,1.62797199916713*pi) q[9];
U1q(0.381649274849055*pi,0.9124702607519901*pi) q[10];
U1q(0.437692619213021*pi,0.8934873052532*pi) q[11];
U1q(0.0721034702653671*pi,0.7314195690466798*pi) q[12];
U1q(0.590749718987806*pi,0.7889953107263201*pi) q[13];
U1q(0.309383606758226*pi,1.06774810825116*pi) q[14];
U1q(0.728015433750618*pi,1.9258160877026*pi) q[15];
U1q(0.393322063844905*pi,1.5156471357661596*pi) q[16];
U1q(0.510002318558666*pi,0.9403765925160501*pi) q[17];
U1q(0.753772831503716*pi,0.10414651830547994*pi) q[18];
U1q(0.197553114534385*pi,1.3306661943622196*pi) q[19];
U1q(0.11280649527742*pi,1.1547374616880104*pi) q[20];
U1q(0.670212271102533*pi,1.8597100272556997*pi) q[21];
U1q(0.184286744484914*pi,1.56502479127378*pi) q[22];
U1q(0.567949068095178*pi,1.9863485736103899*pi) q[23];
U1q(0.665661389675184*pi,0.7356565718717198*pi) q[24];
U1q(0.465485412793492*pi,0.13301209739100006*pi) q[25];
U1q(0.158333360355498*pi,0.19130971728520008*pi) q[26];
U1q(0.305875018854416*pi,0.1752188242039101*pi) q[27];
U1q(0.571282268773575*pi,1.0513338423902496*pi) q[28];
U1q(0.244766185746887*pi,0.44331825648976997*pi) q[29];
U1q(0.695377453328877*pi,1.6708776889452999*pi) q[30];
U1q(0.545554837627028*pi,1.65484010530216*pi) q[31];
U1q(0.502229716754877*pi,0.0949027723634801*pi) q[32];
U1q(0.323572493814354*pi,1.0398259586509102*pi) q[33];
U1q(0.157427781793604*pi,0.8775528148902798*pi) q[34];
U1q(0.230148959214129*pi,0.003988160784389905*pi) q[35];
U1q(0.325310385744688*pi,1.9877461513502102*pi) q[36];
U1q(0.677090270912203*pi,0.06074883980548007*pi) q[37];
U1q(0.228341253570881*pi,0.8900415336061802*pi) q[38];
U1q(0.865826468873657*pi,0.57092726084276*pi) q[39];
U1q(0.467426901481337*pi,0.042248991319029816*pi) q[40];
U1q(0.296353816957645*pi,1.21842789923766*pi) q[41];
U1q(0.92538060987831*pi,1.4314290474564202*pi) q[42];
U1q(0.263959046869744*pi,0.24743764287936987*pi) q[43];
U1q(0.308368513572731*pi,0.5391759985951501*pi) q[44];
U1q(0.581729135142095*pi,1.03154685346139*pi) q[45];
U1q(0.176863910661645*pi,1.7921778423193002*pi) q[46];
U1q(0.59354133013103*pi,1.1393804242524297*pi) q[47];
U1q(0.554779076392504*pi,0.23846981787006993*pi) q[48];
U1q(0.704314192546026*pi,1.4005750049246304*pi) q[49];
U1q(0.932443995255401*pi,0.2933096418579799*pi) q[50];
U1q(0.0254102778634905*pi,1.0744269135387499*pi) q[51];
U1q(0.4854670543387*pi,0.5495045075634204*pi) q[52];
U1q(0.389252527481112*pi,1.63346655552595*pi) q[53];
U1q(0.459856797023776*pi,1.7338034257432202*pi) q[54];
U1q(0.425083668635223*pi,0.19706005854166975*pi) q[55];
RZZ(0.5*pi) q[22],q[0];
RZZ(0.5*pi) q[50],q[1];
RZZ(0.5*pi) q[2],q[39];
RZZ(0.5*pi) q[11],q[3];
RZZ(0.5*pi) q[4],q[43];
RZZ(0.5*pi) q[5],q[53];
RZZ(0.5*pi) q[14],q[6];
RZZ(0.5*pi) q[7],q[25];
RZZ(0.5*pi) q[8],q[19];
RZZ(0.5*pi) q[9],q[16];
RZZ(0.5*pi) q[38],q[10];
RZZ(0.5*pi) q[12],q[36];
RZZ(0.5*pi) q[37],q[13];
RZZ(0.5*pi) q[15],q[52];
RZZ(0.5*pi) q[17],q[47];
RZZ(0.5*pi) q[18],q[48];
RZZ(0.5*pi) q[20],q[33];
RZZ(0.5*pi) q[21],q[35];
RZZ(0.5*pi) q[31],q[23];
RZZ(0.5*pi) q[24],q[32];
RZZ(0.5*pi) q[26],q[46];
RZZ(0.5*pi) q[27],q[29];
RZZ(0.5*pi) q[51],q[28];
RZZ(0.5*pi) q[30],q[41];
RZZ(0.5*pi) q[34],q[54];
RZZ(0.5*pi) q[44],q[40];
RZZ(0.5*pi) q[42],q[45];
RZZ(0.5*pi) q[55],q[49];
U1q(0.752671002547931*pi,0.2269082159714797*pi) q[0];
U1q(0.391693254228537*pi,0.0370308050060002*pi) q[1];
U1q(0.163673822953171*pi,0.04129311710094008*pi) q[2];
U1q(0.39014328362217*pi,1.9480157201163504*pi) q[3];
U1q(0.751759382263858*pi,1.48552112372131*pi) q[4];
U1q(0.42642386959532*pi,0.94213550814891*pi) q[5];
U1q(0.669898101214701*pi,0.4205891618137496*pi) q[6];
U1q(0.293921045361541*pi,0.18109231556235983*pi) q[7];
U1q(0.377445726180446*pi,1.0906967740766396*pi) q[8];
U1q(0.88064120993588*pi,0.3392319596220199*pi) q[9];
U1q(0.834094711522545*pi,0.6315248731477103*pi) q[10];
U1q(0.78072514323134*pi,0.05001748455404975*pi) q[11];
U1q(0.39389327854613*pi,0.9629546415536403*pi) q[12];
U1q(0.213170370798846*pi,0.18294769704875957*pi) q[13];
U1q(0.737316955515104*pi,1.8206283709618702*pi) q[14];
U1q(0.466823575800634*pi,0.9331697288959298*pi) q[15];
U1q(0.7066014269422*pi,1.5943229518674809*pi) q[16];
U1q(0.664565331689154*pi,0.4465321611465898*pi) q[17];
U1q(0.632215293502428*pi,1.0685220345201802*pi) q[18];
U1q(0.427877806392378*pi,1.7449687180115703*pi) q[19];
U1q(0.705359457710585*pi,0.37126800122696046*pi) q[20];
U1q(0.635713982843666*pi,1.7490309670557398*pi) q[21];
U1q(0.272135181338912*pi,1.91276494970999*pi) q[22];
U1q(0.44168750548674*pi,1.6557561914384493*pi) q[23];
U1q(0.516442966896176*pi,1.3178271645017698*pi) q[24];
U1q(0.130461930798006*pi,1.9106707147483402*pi) q[25];
U1q(0.417177138499074*pi,1.5812777696144504*pi) q[26];
U1q(0.367752223388094*pi,0.60644000761254*pi) q[27];
U1q(0.69637167413619*pi,1.73846553924853*pi) q[28];
U1q(0.245397116556629*pi,0.0755439317311799*pi) q[29];
U1q(0.358694842707641*pi,1.0623020716129101*pi) q[30];
U1q(0.794006335341692*pi,0.19502836823041036*pi) q[31];
U1q(0.403649571950172*pi,1.0493956256977697*pi) q[32];
U1q(0.85612796736783*pi,1.0020036079803996*pi) q[33];
U1q(0.23829775363317*pi,1.1730590175255298*pi) q[34];
U1q(0.46618854862123*pi,0.5397169806325195*pi) q[35];
U1q(0.624966971734559*pi,1.2531244371603902*pi) q[36];
U1q(0.554851182317714*pi,0.77036726427021*pi) q[37];
U1q(0.56151920224492*pi,0.1957097340006202*pi) q[38];
U1q(0.179014450082454*pi,0.9723341812224398*pi) q[39];
U1q(0.489106982769057*pi,0.4476438648999901*pi) q[40];
U1q(0.397006403043181*pi,0.07252920897319992*pi) q[41];
U1q(0.21002747464114*pi,1.86023668062347*pi) q[42];
U1q(0.656952174158548*pi,0.7073783248401302*pi) q[43];
U1q(0.624300401681687*pi,0.45028076861991995*pi) q[44];
U1q(0.639960227824079*pi,1.8583360527728292*pi) q[45];
U1q(0.242270045425239*pi,0.5700548105825707*pi) q[46];
U1q(0.592371088051627*pi,1.0966866029894096*pi) q[47];
U1q(0.454151177915925*pi,0.32390490560942986*pi) q[48];
U1q(0.535239308699178*pi,0.7634260867918696*pi) q[49];
U1q(0.403856850625538*pi,0.38697786007258017*pi) q[50];
U1q(0.229821253168404*pi,0.07433270204862996*pi) q[51];
U1q(0.465928582759314*pi,0.44395148814706964*pi) q[52];
U1q(0.512381228538959*pi,0.7510477550402204*pi) q[53];
U1q(0.559017640331486*pi,1.8391388248802203*pi) q[54];
U1q(0.240190275279394*pi,1.8461455056409708*pi) q[55];
RZZ(0.5*pi) q[27],q[0];
RZZ(0.5*pi) q[22],q[1];
RZZ(0.5*pi) q[9],q[2];
RZZ(0.5*pi) q[3],q[19];
RZZ(0.5*pi) q[4],q[5];
RZZ(0.5*pi) q[35],q[6];
RZZ(0.5*pi) q[7],q[46];
RZZ(0.5*pi) q[30],q[8];
RZZ(0.5*pi) q[24],q[10];
RZZ(0.5*pi) q[11],q[43];
RZZ(0.5*pi) q[34],q[12];
RZZ(0.5*pi) q[13],q[44];
RZZ(0.5*pi) q[14],q[54];
RZZ(0.5*pi) q[15],q[31];
RZZ(0.5*pi) q[16],q[33];
RZZ(0.5*pi) q[17],q[53];
RZZ(0.5*pi) q[18],q[28];
RZZ(0.5*pi) q[29],q[20];
RZZ(0.5*pi) q[38],q[21];
RZZ(0.5*pi) q[40],q[23];
RZZ(0.5*pi) q[48],q[25];
RZZ(0.5*pi) q[26],q[37];
RZZ(0.5*pi) q[50],q[32];
RZZ(0.5*pi) q[36],q[41];
RZZ(0.5*pi) q[39],q[52];
RZZ(0.5*pi) q[55],q[42];
RZZ(0.5*pi) q[51],q[45];
RZZ(0.5*pi) q[49],q[47];
U1q(0.165675740497493*pi,0.010071844395609375*pi) q[0];
U1q(0.484308916313454*pi,0.8990859866919703*pi) q[1];
U1q(0.216006712057912*pi,0.2077715221982004*pi) q[2];
U1q(0.274902824422092*pi,1.9228361407586192*pi) q[3];
U1q(0.463941747477974*pi,1.4068601345820095*pi) q[4];
U1q(0.567810783979796*pi,0.6495517024141897*pi) q[5];
U1q(0.909913296698857*pi,0.8872083413522098*pi) q[6];
U1q(0.393628723804495*pi,0.1641186537860797*pi) q[7];
U1q(0.801956097336115*pi,0.3617868607980892*pi) q[8];
U1q(0.160134491737458*pi,1.7226593100162901*pi) q[9];
U1q(0.117101509428526*pi,1.7983908784672398*pi) q[10];
U1q(0.75881950433988*pi,1.7049190699500496*pi) q[11];
U1q(0.467806217679011*pi,1.4857757745689*pi) q[12];
U1q(0.679950066796139*pi,0.20440888427411075*pi) q[13];
U1q(0.751104061483904*pi,0.3645169665329897*pi) q[14];
U1q(0.419009458224554*pi,0.27001407331209926*pi) q[15];
U1q(0.249663333030816*pi,1.6410674539776995*pi) q[16];
U1q(0.378241907823873*pi,1.2816006333325003*pi) q[17];
U1q(0.734769162685723*pi,0.7231191444007408*pi) q[18];
U1q(0.531696203519432*pi,0.44580583535501983*pi) q[19];
U1q(0.454517293632707*pi,0.21330239103120086*pi) q[20];
U1q(0.4732057569566*pi,0.3151042596101501*pi) q[21];
U1q(0.604902401352994*pi,1.1585723723000996*pi) q[22];
U1q(0.521008341408055*pi,0.5925320543971004*pi) q[23];
U1q(0.196422374099983*pi,0.47404250169369977*pi) q[24];
U1q(0.234162010588154*pi,0.7659027786064296*pi) q[25];
U1q(0.650002748972624*pi,0.1568006879311099*pi) q[26];
U1q(0.622918380835626*pi,1.3625548866759702*pi) q[27];
U1q(0.399795483277526*pi,1.0140554414426504*pi) q[28];
U1q(0.587268038878231*pi,1.2868530290916809*pi) q[29];
U1q(0.791683298885783*pi,0.6984861126317101*pi) q[30];
U1q(0.121165404979164*pi,0.9630974162499903*pi) q[31];
U1q(0.900519866771023*pi,0.13005584795459946*pi) q[32];
U1q(0.709202867962852*pi,1.8642947821837996*pi) q[33];
U1q(0.572812738000891*pi,1.3932314838781403*pi) q[34];
U1q(0.16026200328552*pi,0.7572464143621005*pi) q[35];
U1q(0.46093456854409*pi,1.5650589319931196*pi) q[36];
U1q(0.712475499834619*pi,1.5461400412311903*pi) q[37];
U1q(0.640764684483015*pi,0.4937280554011103*pi) q[38];
U1q(0.864527278154715*pi,1.5093690054100009*pi) q[39];
U1q(0.572510073097393*pi,1.3231453786447709*pi) q[40];
U1q(0.486604109600428*pi,1.6597572579174198*pi) q[41];
U1q(0.02854125631068*pi,0.5457824437062397*pi) q[42];
U1q(0.333735898296031*pi,1.6850596176521009*pi) q[43];
U1q(0.804799125726132*pi,0.11056533104116006*pi) q[44];
U1q(0.334196754113127*pi,0.0663975555415508*pi) q[45];
U1q(0.586486093698969*pi,1.7664098150328993*pi) q[46];
U1q(0.524881035518385*pi,1.3403980604222099*pi) q[47];
U1q(0.570758272197956*pi,1.2648032432759804*pi) q[48];
U1q(0.5642053782469*pi,1.00901652608448*pi) q[49];
U1q(0.19896034487162*pi,0.8454560192130698*pi) q[50];
U1q(0.324997452976324*pi,0.7796619989189004*pi) q[51];
U1q(0.400606908304153*pi,1.2567512600660002*pi) q[52];
U1q(0.54553161327213*pi,1.6019174754573005*pi) q[53];
U1q(0.683715582925262*pi,1.5942299171504999*pi) q[54];
U1q(0.449195306911767*pi,1.0125368197992994*pi) q[55];
RZZ(0.5*pi) q[31],q[0];
RZZ(0.5*pi) q[39],q[1];
RZZ(0.5*pi) q[2],q[25];
RZZ(0.5*pi) q[3],q[23];
RZZ(0.5*pi) q[4],q[41];
RZZ(0.5*pi) q[9],q[5];
RZZ(0.5*pi) q[16],q[6];
RZZ(0.5*pi) q[7],q[45];
RZZ(0.5*pi) q[20],q[8];
RZZ(0.5*pi) q[12],q[10];
RZZ(0.5*pi) q[11],q[14];
RZZ(0.5*pi) q[13],q[52];
RZZ(0.5*pi) q[15],q[44];
RZZ(0.5*pi) q[17],q[24];
RZZ(0.5*pi) q[18],q[36];
RZZ(0.5*pi) q[26],q[19];
RZZ(0.5*pi) q[21],q[55];
RZZ(0.5*pi) q[50],q[22];
RZZ(0.5*pi) q[27],q[53];
RZZ(0.5*pi) q[48],q[28];
RZZ(0.5*pi) q[29],q[40];
RZZ(0.5*pi) q[30],q[37];
RZZ(0.5*pi) q[32],q[34];
RZZ(0.5*pi) q[46],q[33];
RZZ(0.5*pi) q[38],q[35];
RZZ(0.5*pi) q[49],q[42];
RZZ(0.5*pi) q[43],q[47];
RZZ(0.5*pi) q[51],q[54];
U1q(0.225719155478599*pi,0.7608257015319992*pi) q[0];
U1q(0.228153805276832*pi,0.4735326307944998*pi) q[1];
U1q(0.11844228042274*pi,0.7264886672024993*pi) q[2];
U1q(0.519885066683214*pi,0.7247763602771098*pi) q[3];
U1q(0.734974567358488*pi,0.7273116602776604*pi) q[4];
U1q(0.800796674332851*pi,0.29176090850010006*pi) q[5];
U1q(0.723126140926772*pi,1.5514512348248104*pi) q[6];
U1q(0.568259343046769*pi,1.5844180265768006*pi) q[7];
U1q(0.757572835105946*pi,0.45415667255210046*pi) q[8];
U1q(0.415389424717362*pi,0.7179649292713002*pi) q[9];
U1q(0.640923256329866*pi,1.4930462658462993*pi) q[10];
U1q(0.782471436741081*pi,1.8996884372613003*pi) q[11];
U1q(0.557377191935049*pi,0.3109282844086003*pi) q[12];
U1q(0.407816047223538*pi,0.4658644288576994*pi) q[13];
U1q(0.828497700742839*pi,1.51009766523598*pi) q[14];
U1q(0.584422863407477*pi,0.40425721566010075*pi) q[15];
U1q(0.495275210019424*pi,1.4983538159227*pi) q[16];
U1q(0.535186229362659*pi,0.4516532604447008*pi) q[17];
U1q(0.186270344191765*pi,1.1437272741660003*pi) q[18];
U1q(0.742829787869133*pi,0.34548391564477043*pi) q[19];
U1q(0.764838835227068*pi,0.9908745107154999*pi) q[20];
U1q(0.577609432596102*pi,0.2835732812584304*pi) q[21];
U1q(0.0708461728331632*pi,1.3222850567258*pi) q[22];
U1q(0.500671013145457*pi,1.6687534297764994*pi) q[23];
U1q(0.442020108621628*pi,1.7275523012561003*pi) q[24];
U1q(0.383631509150583*pi,0.9245623518133996*pi) q[25];
U1q(0.378118776699904*pi,1.0537595066544991*pi) q[26];
U1q(0.13199044239442*pi,1.80367340945093*pi) q[27];
U1q(0.441840683236462*pi,0.5325696265963007*pi) q[28];
U1q(0.667992833347366*pi,1.8818411116060005*pi) q[29];
U1q(0.57141892664851*pi,0.6262733921342996*pi) q[30];
U1q(0.343670829853332*pi,1.8639261019645996*pi) q[31];
U1q(0.980712424177744*pi,1.6849966156638008*pi) q[32];
U1q(0.217231271217126*pi,0.5237371071286994*pi) q[33];
U1q(0.780989037253882*pi,0.24825085507506994*pi) q[34];
U1q(0.179585368905214*pi,0.3638482732712003*pi) q[35];
U1q(0.833019964629009*pi,1.2881211300085003*pi) q[36];
U1q(0.481333720353128*pi,1.6022096197951203*pi) q[37];
U1q(0.646816411158209*pi,0.4148682234727996*pi) q[38];
U1q(0.615529331146837*pi,1.4445390136438991*pi) q[39];
U1q(0.765076534898975*pi,0.9326031425439005*pi) q[40];
U1q(0.591994721422576*pi,0.1434458390163993*pi) q[41];
U1q(0.570791006979326*pi,0.3855686402796099*pi) q[42];
U1q(0.443941777843626*pi,1.1565574211089*pi) q[43];
U1q(0.810740969129256*pi,1.6305806695603504*pi) q[44];
U1q(0.413608056500387*pi,1.7983640763207998*pi) q[45];
U1q(0.734356390336656*pi,1.2260996923594991*pi) q[46];
U1q(0.219998124875949*pi,0.5446595535685006*pi) q[47];
U1q(0.174643749530362*pi,0.26671757323246936*pi) q[48];
U1q(0.61945328033217*pi,1.1148997025961993*pi) q[49];
U1q(0.0455717266229572*pi,0.8763772630076501*pi) q[50];
U1q(0.162935624355565*pi,0.9612964237983999*pi) q[51];
U1q(0.0992667387537241*pi,1.7335023361466*pi) q[52];
U1q(0.261237172640867*pi,1.1797153678885*pi) q[53];
U1q(0.400703406045466*pi,0.6697743078096003*pi) q[54];
U1q(0.872773735259587*pi,1.8925748816969996*pi) q[55];
RZZ(0.5*pi) q[41],q[0];
RZZ(0.5*pi) q[32],q[1];
RZZ(0.5*pi) q[2],q[19];
RZZ(0.5*pi) q[35],q[3];
RZZ(0.5*pi) q[7],q[4];
RZZ(0.5*pi) q[5],q[43];
RZZ(0.5*pi) q[25],q[6];
RZZ(0.5*pi) q[8],q[37];
RZZ(0.5*pi) q[9],q[36];
RZZ(0.5*pi) q[18],q[10];
RZZ(0.5*pi) q[26],q[11];
RZZ(0.5*pi) q[12],q[55];
RZZ(0.5*pi) q[13],q[23];
RZZ(0.5*pi) q[14],q[53];
RZZ(0.5*pi) q[15],q[54];
RZZ(0.5*pi) q[34],q[16];
RZZ(0.5*pi) q[17],q[20];
RZZ(0.5*pi) q[51],q[21];
RZZ(0.5*pi) q[22],q[40];
RZZ(0.5*pi) q[24],q[49];
RZZ(0.5*pi) q[27],q[38];
RZZ(0.5*pi) q[29],q[28];
RZZ(0.5*pi) q[30],q[48];
RZZ(0.5*pi) q[31],q[44];
RZZ(0.5*pi) q[52],q[33];
RZZ(0.5*pi) q[50],q[39];
RZZ(0.5*pi) q[47],q[42];
RZZ(0.5*pi) q[46],q[45];
U1q(0.130038226707374*pi,1.4564517479494015*pi) q[0];
U1q(0.320544680642958*pi,0.09619810721270028*pi) q[1];
U1q(0.420637197580239*pi,0.9935792660667992*pi) q[2];
U1q(0.47910836059573*pi,1.8244416987446002*pi) q[3];
U1q(0.443127597561128*pi,1.5263467870212004*pi) q[4];
U1q(0.201540678402616*pi,0.7644973673125008*pi) q[5];
U1q(0.146846999837415*pi,0.27866635228549974*pi) q[6];
U1q(0.534947685942059*pi,0.40637639697539996*pi) q[7];
U1q(0.79176389819035*pi,1.5569815438215997*pi) q[8];
U1q(0.657444761348163*pi,1.9047880202315*pi) q[9];
U1q(0.631428330804204*pi,0.4041033139583998*pi) q[10];
U1q(0.415076952089273*pi,0.4843935218670996*pi) q[11];
U1q(0.726501349475396*pi,0.07179394228879943*pi) q[12];
U1q(0.643071554065953*pi,1.8068390138762993*pi) q[13];
U1q(0.306705327824834*pi,1.6354356041630993*pi) q[14];
U1q(0.379971277373442*pi,1.6270087520959997*pi) q[15];
U1q(0.407061533212483*pi,1.0018231448284993*pi) q[16];
U1q(0.353010407345166*pi,1.0921984933930986*pi) q[17];
U1q(0.797865791799741*pi,0.7847473727050005*pi) q[18];
U1q(0.35687829679239*pi,1.6480015528749998*pi) q[19];
U1q(0.376331233855114*pi,0.5246102398488013*pi) q[20];
U1q(0.469667229898736*pi,1.2731510353717006*pi) q[21];
U1q(0.728411947030602*pi,1.4335219758792999*pi) q[22];
U1q(0.184444959060579*pi,0.7347069850903996*pi) q[23];
U1q(0.678429184530667*pi,0.5387769418213999*pi) q[24];
U1q(0.648875342967261*pi,0.6419989401475004*pi) q[25];
U1q(0.759868611684525*pi,1.4783468746812005*pi) q[26];
U1q(0.41323248248024*pi,1.9463871165204996*pi) q[27];
U1q(0.239977779623747*pi,1.7609480583543995*pi) q[28];
U1q(0.577567705591836*pi,1.7141593792863006*pi) q[29];
U1q(0.69838643024271*pi,0.8550111230848998*pi) q[30];
U1q(0.6199719116027*pi,0.4834754338747995*pi) q[31];
U1q(0.168710356262442*pi,0.1367244884719998*pi) q[32];
U1q(0.454791356525738*pi,1.8533527509099983*pi) q[33];
U1q(0.73364735340071*pi,0.7216045001729299*pi) q[34];
U1q(0.309576162927621*pi,1.9382696839886009*pi) q[35];
U1q(0.305029105340111*pi,1.4517624422701*pi) q[36];
U1q(0.106008095939831*pi,0.5602971963780998*pi) q[37];
U1q(0.25837802301084*pi,0.3421903640193005*pi) q[38];
U1q(0.60951551105131*pi,0.15733959908139994*pi) q[39];
U1q(0.369112587205335*pi,0.4964335385168006*pi) q[40];
U1q(0.691960309291721*pi,0.3974224419414991*pi) q[41];
U1q(0.556101526723329*pi,1.4295957006479991*pi) q[42];
U1q(0.215473489842218*pi,0.2888024630231989*pi) q[43];
U1q(0.584213174161601*pi,1.31281649089118*pi) q[44];
U1q(0.493189974795753*pi,0.9706961759815993*pi) q[45];
U1q(0.768870004628812*pi,0.6788236736610003*pi) q[46];
U1q(0.221320434762291*pi,1.5787510320620992*pi) q[47];
U1q(0.758001077734899*pi,1.9264031001220001*pi) q[48];
U1q(0.756335614466964*pi,0.3417912360587003*pi) q[49];
U1q(0.539612590656516*pi,1.4137293389074994*pi) q[50];
U1q(0.172518855188718*pi,1.5044706302609008*pi) q[51];
U1q(0.588690142023829*pi,1.2635485978664995*pi) q[52];
U1q(0.169382426340669*pi,0.3762554972940997*pi) q[53];
U1q(0.518260620586493*pi,1.8812311178959007*pi) q[54];
U1q(0.498005712075653*pi,1.8952915447449996*pi) q[55];
RZZ(0.5*pi) q[25],q[0];
RZZ(0.5*pi) q[41],q[1];
RZZ(0.5*pi) q[27],q[2];
RZZ(0.5*pi) q[7],q[3];
RZZ(0.5*pi) q[4],q[9];
RZZ(0.5*pi) q[5],q[6];
RZZ(0.5*pi) q[8],q[23];
RZZ(0.5*pi) q[10],q[35];
RZZ(0.5*pi) q[18],q[11];
RZZ(0.5*pi) q[29],q[12];
RZZ(0.5*pi) q[13],q[53];
RZZ(0.5*pi) q[14],q[45];
RZZ(0.5*pi) q[15],q[30];
RZZ(0.5*pi) q[36],q[16];
RZZ(0.5*pi) q[17],q[54];
RZZ(0.5*pi) q[49],q[19];
RZZ(0.5*pi) q[20],q[43];
RZZ(0.5*pi) q[21],q[42];
RZZ(0.5*pi) q[46],q[22];
RZZ(0.5*pi) q[24],q[48];
RZZ(0.5*pi) q[26],q[32];
RZZ(0.5*pi) q[28],q[33];
RZZ(0.5*pi) q[39],q[31];
RZZ(0.5*pi) q[34],q[37];
RZZ(0.5*pi) q[38],q[52];
RZZ(0.5*pi) q[51],q[40];
RZZ(0.5*pi) q[55],q[44];
RZZ(0.5*pi) q[50],q[47];
U1q(0.193777120621218*pi,1.5687752013343008*pi) q[0];
U1q(0.189747198525316*pi,1.2560294743648992*pi) q[1];
U1q(0.384668039453849*pi,0.22486710500299978*pi) q[2];
U1q(0.469337754012916*pi,1.2231579549724998*pi) q[3];
U1q(0.441584025141596*pi,0.26797570956479966*pi) q[4];
U1q(0.543890155400878*pi,1.4166744050430005*pi) q[5];
U1q(0.325600199439111*pi,1.4062697816740997*pi) q[6];
U1q(0.579087827097445*pi,0.5544423665752998*pi) q[7];
U1q(0.837360765177473*pi,1.0880435823731993*pi) q[8];
U1q(0.540398476729862*pi,1.638315130860601*pi) q[9];
U1q(0.781857157857144*pi,1.4346927848744997*pi) q[10];
U1q(0.46171527761216*pi,0.3080246478230997*pi) q[11];
U1q(0.371063960496578*pi,1.6557743273221988*pi) q[12];
U1q(0.201139618843071*pi,0.31439690757409977*pi) q[13];
U1q(0.375572326563624*pi,0.45696487858820056*pi) q[14];
U1q(0.647180102745018*pi,0.27155752909190056*pi) q[15];
U1q(0.715842709441203*pi,1.6707810215233998*pi) q[16];
U1q(0.644822314911007*pi,1.6181620220725996*pi) q[17];
U1q(0.653559543613911*pi,1.9571194499511009*pi) q[18];
U1q(0.541207147714098*pi,0.6273922042894*pi) q[19];
U1q(0.487154078943304*pi,0.8815198636870996*pi) q[20];
U1q(0.228742922479513*pi,1.4129077135348993*pi) q[21];
U1q(0.232728483059053*pi,1.1216064125312002*pi) q[22];
U1q(0.460334169285826*pi,1.9321722377252009*pi) q[23];
U1q(0.464464477688365*pi,0.5038543305687*pi) q[24];
U1q(0.109165337017199*pi,1.9281496201082007*pi) q[25];
U1q(0.195598454171182*pi,0.1735973068901*pi) q[26];
U1q(0.627203063563348*pi,0.7019235798907992*pi) q[27];
U1q(0.661161055664502*pi,1.1381437152248992*pi) q[28];
U1q(0.755353966536609*pi,1.5098556007066009*pi) q[29];
U1q(0.341547286710637*pi,1.1008647469169013*pi) q[30];
U1q(0.703186458465648*pi,1.6700199259971988*pi) q[31];
U1q(0.462389510262013*pi,1.5521686256949003*pi) q[32];
U1q(0.486592811257384*pi,0.9923080588420987*pi) q[33];
U1q(0.705366175063755*pi,0.7455915881932*pi) q[34];
U1q(0.725509097021671*pi,1.1835775535551996*pi) q[35];
U1q(0.592772804403463*pi,1.3869716964380991*pi) q[36];
U1q(0.0956330564390971*pi,0.9249115428929002*pi) q[37];
U1q(0.776318251420946*pi,0.9607335568426993*pi) q[38];
U1q(0.126558189259684*pi,0.8058470214382005*pi) q[39];
U1q(0.274159078850449*pi,1.1927081765080985*pi) q[40];
U1q(0.421038797400106*pi,0.14148220853330074*pi) q[41];
U1q(0.491652012925096*pi,1.9437430863935*pi) q[42];
U1q(0.257174775815042*pi,1.4132415730826011*pi) q[43];
U1q(0.646729591118289*pi,1.0787192619953991*pi) q[44];
U1q(0.724892485932104*pi,1.6908125048248017*pi) q[45];
U1q(0.482589159982332*pi,0.9577749841682994*pi) q[46];
U1q(0.726173045266135*pi,1.0835433041320996*pi) q[47];
U1q(0.718829216486527*pi,1.9091872690848*pi) q[48];
U1q(0.810133735525596*pi,0.6004615494447005*pi) q[49];
U1q(0.297616076414669*pi,1.6785487281391003*pi) q[50];
U1q(0.312345807614835*pi,0.8379384006675998*pi) q[51];
U1q(0.836083717221631*pi,0.2525762045838995*pi) q[52];
U1q(0.594859631178454*pi,0.5270567514411013*pi) q[53];
U1q(0.404028101035177*pi,0.8997113610005982*pi) q[54];
U1q(0.577532742953899*pi,0.9539954167547009*pi) q[55];
RZZ(0.5*pi) q[39],q[0];
RZZ(0.5*pi) q[11],q[1];
RZZ(0.5*pi) q[2],q[31];
RZZ(0.5*pi) q[55],q[3];
RZZ(0.5*pi) q[4],q[24];
RZZ(0.5*pi) q[22],q[5];
RZZ(0.5*pi) q[28],q[6];
RZZ(0.5*pi) q[7],q[34];
RZZ(0.5*pi) q[38],q[8];
RZZ(0.5*pi) q[9],q[14];
RZZ(0.5*pi) q[13],q[10];
RZZ(0.5*pi) q[17],q[12];
RZZ(0.5*pi) q[15],q[35];
RZZ(0.5*pi) q[37],q[16];
RZZ(0.5*pi) q[18],q[53];
RZZ(0.5*pi) q[51],q[19];
RZZ(0.5*pi) q[20],q[21];
RZZ(0.5*pi) q[50],q[23];
RZZ(0.5*pi) q[49],q[25];
RZZ(0.5*pi) q[30],q[26];
RZZ(0.5*pi) q[27],q[42];
RZZ(0.5*pi) q[29],q[52];
RZZ(0.5*pi) q[32],q[48];
RZZ(0.5*pi) q[33],q[43];
RZZ(0.5*pi) q[36],q[46];
RZZ(0.5*pi) q[40],q[54];
RZZ(0.5*pi) q[44],q[41];
RZZ(0.5*pi) q[47],q[45];
U1q(0.809737364639506*pi,0.36984083734479967*pi) q[0];
U1q(0.375225148249758*pi,0.7372495082890005*pi) q[1];
U1q(0.695264020160273*pi,1.8594436557472989*pi) q[2];
U1q(0.739975452411648*pi,1.5401220643120013*pi) q[3];
U1q(0.553241580408469*pi,0.8534402752367996*pi) q[4];
U1q(0.473212857975276*pi,1.0073080492078006*pi) q[5];
U1q(0.545979161794372*pi,1.9467922497054992*pi) q[6];
U1q(0.773401242152623*pi,1.5815977702430004*pi) q[7];
U1q(0.534927045159229*pi,0.35030776192660085*pi) q[8];
U1q(0.449971732637466*pi,1.281685084586801*pi) q[9];
U1q(0.71329623961774*pi,1.5265052095915017*pi) q[10];
U1q(0.209217190284786*pi,1.3002440913152*pi) q[11];
U1q(0.178777620736355*pi,0.7571033508484994*pi) q[12];
U1q(0.534232876821122*pi,1.0444943840897984*pi) q[13];
U1q(0.474229517254273*pi,0.48697279471610067*pi) q[14];
U1q(0.710005695292322*pi,0.05156193489850125*pi) q[15];
U1q(0.537239199409604*pi,1.1641542806783*pi) q[16];
U1q(0.345049160069171*pi,0.6399624179423*pi) q[17];
U1q(0.542922561812811*pi,0.4146055034970004*pi) q[18];
U1q(0.34364354166175*pi,0.4806511892181007*pi) q[19];
U1q(0.342347838283488*pi,0.7772179464267985*pi) q[20];
U1q(0.0682978773443886*pi,0.3870575208125011*pi) q[21];
U1q(0.279165964440343*pi,0.7289412229884995*pi) q[22];
U1q(0.737043567435075*pi,0.014248793679698224*pi) q[23];
U1q(0.261586006390571*pi,0.8213474320537983*pi) q[24];
U1q(0.688452095221924*pi,1.757474566591899*pi) q[25];
U1q(0.3896771040453*pi,0.6040834457551014*pi) q[26];
U1q(0.510257454290998*pi,1.787116498782499*pi) q[27];
U1q(0.588164971017664*pi,1.2394726487631011*pi) q[28];
U1q(0.877963560284115*pi,1.0140895763769997*pi) q[29];
U1q(0.207496278399317*pi,0.8776604012670006*pi) q[30];
U1q(0.279371467923501*pi,0.11152605945850169*pi) q[31];
U1q(0.497729473794263*pi,1.6422300902466986*pi) q[32];
U1q(0.108295031819174*pi,1.3583459591509985*pi) q[33];
U1q(0.625655275076477*pi,0.5088968288931994*pi) q[34];
U1q(0.276638017573142*pi,0.5169310692069011*pi) q[35];
U1q(0.361305772529675*pi,0.5567180404338998*pi) q[36];
U1q(0.798760102892233*pi,0.4516447702383992*pi) q[37];
U1q(0.89594380701152*pi,0.0034509636788992992*pi) q[38];
U1q(0.0732730865103041*pi,0.6543346678767996*pi) q[39];
U1q(0.8686342754255*pi,0.22404823383449823*pi) q[40];
U1q(0.143807684592179*pi,1.8264703246817007*pi) q[41];
U1q(0.797585047392648*pi,1.8562066003691005*pi) q[42];
U1q(0.251887673602385*pi,1.5923093245900013*pi) q[43];
U1q(0.706784388422684*pi,0.4802142136610996*pi) q[44];
U1q(0.0437111119372165*pi,1.3471075323478985*pi) q[45];
U1q(0.524183445772233*pi,0.33483474891890097*pi) q[46];
U1q(0.499486287035912*pi,0.5399899240906016*pi) q[47];
U1q(0.4607522777053*pi,1.9224868872551006*pi) q[48];
U1q(0.153236246407031*pi,1.8019011555011986*pi) q[49];
U1q(0.410454688474085*pi,1.9614725029985998*pi) q[50];
U1q(0.594501155029876*pi,1.3683215564077003*pi) q[51];
U1q(0.643556201650414*pi,1.6066517217236012*pi) q[52];
U1q(0.911593294659872*pi,0.26911568283719944*pi) q[53];
U1q(0.159811319666097*pi,0.3726491841875017*pi) q[54];
U1q(0.556668610967763*pi,1.9046814878208984*pi) q[55];
rz(3.0344904148660987*pi) q[0];
rz(1.3976123193965009*pi) q[1];
rz(3.2941821182168987*pi) q[2];
rz(3.9097739912271*pi) q[3];
rz(3.0832603990701006*pi) q[4];
rz(3.2869675270057996*pi) q[5];
rz(0.6083224341656006*pi) q[6];
rz(2.2726570364274004*pi) q[7];
rz(2.4863871697668998*pi) q[8];
rz(2.4529469052687993*pi) q[9];
rz(0.9346508735042001*pi) q[10];
rz(0.39463395449639904*pi) q[11];
rz(3.3485070571275983*pi) q[12];
rz(3.736845578933899*pi) q[13];
rz(2.232021282379799*pi) q[14];
rz(3.161273220173399*pi) q[15];
rz(3.328077440153301*pi) q[16];
rz(3.3476146028461997*pi) q[17];
rz(0.17409472105750012*pi) q[18];
rz(2.7980688165137018*pi) q[19];
rz(1.8194732337356996*pi) q[20];
rz(3.0609886680621017*pi) q[21];
rz(3.931549334511601*pi) q[22];
rz(2.3079048911301996*pi) q[23];
rz(1.1925098154733007*pi) q[24];
rz(0.23078631198880117*pi) q[25];
rz(0.7601912989393007*pi) q[26];
rz(0.5759538200689001*pi) q[27];
rz(0.12036242848819967*pi) q[28];
rz(3.4210915411943006*pi) q[29];
rz(0.44549282944799984*pi) q[30];
rz(3.1980857290024005*pi) q[31];
rz(0.8150015506828012*pi) q[32];
rz(2.0043832819004983*pi) q[33];
rz(2.6720931559961*pi) q[34];
rz(1.200851144971299*pi) q[35];
rz(0.4698794077593007*pi) q[36];
rz(2.5761644093213008*pi) q[37];
rz(3.5787629056961006*pi) q[38];
rz(0.9308035339781*pi) q[39];
rz(0.6304428385582987*pi) q[40];
rz(1.9529231175258985*pi) q[41];
rz(2.1181994201006002*pi) q[42];
rz(3.5498453549606985*pi) q[43];
rz(0.8047512194953992*pi) q[44];
rz(0.4356952268644001*pi) q[45];
rz(0.4938318000554993*pi) q[46];
rz(2.0017937398921006*pi) q[47];
rz(3.0120129644597995*pi) q[48];
rz(3.334850391144*pi) q[49];
rz(3.0713747746235*pi) q[50];
rz(0.6835062997993013*pi) q[51];
rz(2.7317749731663987*pi) q[52];
rz(3.2021178125368017*pi) q[53];
rz(3.1781342174920013*pi) q[54];
rz(3.433488628133599*pi) q[55];
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
measure q[40] -> c[40];
measure q[41] -> c[41];
measure q[42] -> c[42];
measure q[43] -> c[43];
measure q[44] -> c[44];
measure q[45] -> c[45];
measure q[46] -> c[46];
measure q[47] -> c[47];
measure q[48] -> c[48];
measure q[49] -> c[49];
measure q[50] -> c[50];
measure q[51] -> c[51];
measure q[52] -> c[52];
measure q[53] -> c[53];
measure q[54] -> c[54];
measure q[55] -> c[55];
