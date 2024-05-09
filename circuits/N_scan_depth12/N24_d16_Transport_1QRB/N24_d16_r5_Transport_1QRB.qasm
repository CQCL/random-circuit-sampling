OPENQASM 2.0;
include "hqslib1.inc";

qreg q[24];
creg c[24];
rz(0.318521205294328*pi) q[0];
rz(1.177099717539299*pi) q[1];
rz(0.047108360188474085*pi) q[2];
rz(2.87003800407025*pi) q[3];
rz(0.3897059609334015*pi) q[4];
rz(1.014409506577353*pi) q[5];
rz(0.88481966352701*pi) q[6];
rz(0.618285556076485*pi) q[7];
rz(3.149369633307447*pi) q[8];
rz(3.748668162879003*pi) q[9];
rz(1.90037660958433*pi) q[10];
rz(3.441651099006892*pi) q[11];
rz(0.792820304947536*pi) q[12];
rz(2.8640582423658003*pi) q[13];
rz(2.67195569335317*pi) q[14];
rz(3.500734838975043*pi) q[15];
rz(0.82131996917221*pi) q[16];
rz(0.4674644499059174*pi) q[17];
rz(0.718927022992006*pi) q[18];
rz(3.3374692955265983*pi) q[19];
rz(1.63518636084178*pi) q[20];
rz(1.63782229110087*pi) q[21];
rz(3.4835579272455606*pi) q[22];
rz(1.5637736129504918*pi) q[23];
U1q(0.327133196172385*pi,0.788664953489943*pi) q[0];
U1q(1.44912297741556*pi,1.07049813822157*pi) q[1];
U1q(1.41360267072669*pi,0.618009622547438*pi) q[2];
U1q(1.39678510424639*pi,1.831249730279168*pi) q[3];
U1q(3.17930317498048*pi,1.596827833435455*pi) q[4];
U1q(1.92124996565858*pi,0.343995515896568*pi) q[5];
U1q(1.47218749624982*pi,0.277928440924727*pi) q[6];
U1q(3.430121383663649*pi,1.743110750273962*pi) q[7];
U1q(0.847426347148903*pi,1.547556896788773*pi) q[8];
U1q(0.680178600935393*pi,0.616497456645033*pi) q[9];
U1q(0.548514078999608*pi,1.38953598476487*pi) q[10];
U1q(1.24495527272679*pi,0.232461909326996*pi) q[11];
U1q(0.271070113207642*pi,0.437579559435774*pi) q[12];
U1q(0.880858836355532*pi,1.376770372448058*pi) q[13];
U1q(0.938511979868966*pi,1.262810628220904*pi) q[14];
U1q(1.73856118853482*pi,0.462588075722979*pi) q[15];
U1q(1.65683616399178*pi,1.9552349990497964*pi) q[16];
U1q(1.5831382088525*pi,1.957295500274456*pi) q[17];
U1q(0.0554799754752118*pi,1.12632672735791*pi) q[18];
U1q(3.9124980225339576*pi,1.842548157746209*pi) q[19];
U1q(0.693644928820292*pi,1.49409592518346*pi) q[20];
U1q(0.513976417642434*pi,0.733043834050009*pi) q[21];
U1q(3.746013029765995*pi,0.363196048579127*pi) q[22];
U1q(1.1804507311798*pi,1.46859965143613*pi) q[23];
RZZ(0.0*pi) q[0],q[18];
RZZ(0.0*pi) q[1],q[12];
RZZ(0.0*pi) q[22],q[2];
RZZ(0.0*pi) q[3],q[7];
RZZ(0.0*pi) q[4],q[5];
RZZ(0.0*pi) q[21],q[6];
RZZ(0.0*pi) q[8],q[20];
RZZ(0.0*pi) q[9],q[14];
RZZ(0.0*pi) q[10],q[11];
RZZ(0.0*pi) q[13],q[19];
RZZ(0.0*pi) q[15],q[16];
RZZ(0.0*pi) q[17],q[23];
rz(3.697769106372738*pi) q[0];
rz(3.0778670307763027*pi) q[1];
rz(3.872128149685527*pi) q[2];
rz(2.20091511198092*pi) q[3];
rz(3.663103367141627*pi) q[4];
rz(3.785802057520915*pi) q[5];
rz(2.69375713721167*pi) q[6];
rz(0.842548246433096*pi) q[7];
rz(1.14256650189048*pi) q[8];
rz(3.730833213312581*pi) q[9];
rz(0.455657825013873*pi) q[10];
rz(3.971185183314303*pi) q[11];
rz(0.838537440563918*pi) q[12];
rz(0.662887324326564*pi) q[13];
rz(1.49369217953014*pi) q[14];
rz(0.608195034013732*pi) q[15];
rz(1.23249580275678*pi) q[16];
rz(2.43641026460494*pi) q[17];
rz(1.3184970166463*pi) q[18];
rz(1.62215341650113*pi) q[19];
rz(0.911024223174479*pi) q[20];
rz(3.928137949590562*pi) q[21];
rz(0.682974327184205*pi) q[22];
rz(0.440936846223923*pi) q[23];
U1q(0.0932296347481461*pi,0.569594301566605*pi) q[0];
U1q(0.845624752101647*pi,1.381990499224976*pi) q[1];
U1q(0.439752199814741*pi,1.506332768229448*pi) q[2];
U1q(0.623030390467308*pi,1.565121497072808*pi) q[3];
U1q(0.426321763092566*pi,1.177311363399371*pi) q[4];
U1q(0.849783433886537*pi,0.451791171107989*pi) q[5];
U1q(0.55678779966601*pi,1.42615202266961*pi) q[6];
U1q(0.1693949740817*pi,0.944264871434007*pi) q[7];
U1q(0.204456356359222*pi,0.0286244127957335*pi) q[8];
U1q(0.269371376971701*pi,0.93325474856496*pi) q[9];
U1q(0.990296225820002*pi,0.970758611127979*pi) q[10];
U1q(0.203289141054663*pi,0.465853933920173*pi) q[11];
U1q(0.266883172595946*pi,0.717524063099072*pi) q[12];
U1q(0.291962751344679*pi,1.897849164297081*pi) q[13];
U1q(0.392251881192373*pi,0.827213409065632*pi) q[14];
U1q(0.495113601857425*pi,1.534908319013371*pi) q[15];
U1q(0.258390971150029*pi,1.18336668225818*pi) q[16];
U1q(0.677687437094383*pi,1.579698314109998*pi) q[17];
U1q(0.943780571176651*pi,0.806673367965475*pi) q[18];
U1q(0.723670849492657*pi,1.28192752073202*pi) q[19];
U1q(0.63263003522986*pi,0.573773282245209*pi) q[20];
U1q(0.456791801870755*pi,0.179144979122759*pi) q[21];
U1q(0.625677592130761*pi,0.341094363173681*pi) q[22];
U1q(0.120465688771303*pi,1.802688066606446*pi) q[23];
RZZ(0.0*pi) q[3],q[0];
RZZ(0.0*pi) q[11],q[1];
RZZ(0.0*pi) q[10],q[2];
RZZ(0.0*pi) q[4],q[19];
RZZ(0.0*pi) q[5],q[16];
RZZ(0.0*pi) q[22],q[6];
RZZ(0.0*pi) q[14],q[7];
RZZ(0.0*pi) q[21],q[8];
RZZ(0.0*pi) q[9],q[13];
RZZ(0.0*pi) q[20],q[12];
RZZ(0.0*pi) q[15],q[23];
RZZ(0.0*pi) q[17],q[18];
rz(1.09024092174127*pi) q[0];
rz(1.03632391672526*pi) q[1];
rz(0.865114482844932*pi) q[2];
rz(1.14153475852784*pi) q[3];
rz(3.9335000464709275*pi) q[4];
rz(1.11631718919857*pi) q[5];
rz(0.827561194650689*pi) q[6];
rz(3.633057719087818*pi) q[7];
rz(0.697266638838981*pi) q[8];
rz(0.480975426546315*pi) q[9];
rz(3.523171112350772*pi) q[10];
rz(2.63048824799312*pi) q[11];
rz(0.473100996309619*pi) q[12];
rz(3.77667659842745*pi) q[13];
rz(1.46418921830539*pi) q[14];
rz(0.177097858826834*pi) q[15];
rz(3.7852277019140628*pi) q[16];
rz(1.55080214171998*pi) q[17];
rz(3.608054996654942*pi) q[18];
rz(2.64166784311023*pi) q[19];
rz(2.04091386633314*pi) q[20];
rz(1.49581829942405*pi) q[21];
rz(3.763962380382678*pi) q[22];
rz(3.8854479352754*pi) q[23];
U1q(0.329510724973712*pi,0.742646061640894*pi) q[0];
U1q(0.658386854314745*pi,0.962021325004466*pi) q[1];
U1q(0.379011681380403*pi,1.27953934245613*pi) q[2];
U1q(0.717292270292507*pi,0.778391110098043*pi) q[3];
U1q(0.523747005717486*pi,0.377101901256847*pi) q[4];
U1q(0.770709669120248*pi,0.73413504067686*pi) q[5];
U1q(0.281135077496338*pi,0.555276160622722*pi) q[6];
U1q(0.592421591723399*pi,1.865117477533942*pi) q[7];
U1q(0.475869090275996*pi,0.0851886644337578*pi) q[8];
U1q(0.520183550134924*pi,0.255607246014492*pi) q[9];
U1q(0.518406202801829*pi,0.0356251287819608*pi) q[10];
U1q(0.519520580281111*pi,1.307292839751256*pi) q[11];
U1q(0.569990510336354*pi,0.0522301030669794*pi) q[12];
U1q(0.518235762384854*pi,1.088858418644747*pi) q[13];
U1q(0.651650369763938*pi,1.30822753324305*pi) q[14];
U1q(0.732577546181698*pi,0.827079550848639*pi) q[15];
U1q(0.23240798315264*pi,0.0107672867952663*pi) q[16];
U1q(0.732753533442864*pi,0.946128874870884*pi) q[17];
U1q(0.334080986631069*pi,1.22332176103047*pi) q[18];
U1q(0.642214776967431*pi,1.243547586692112*pi) q[19];
U1q(0.531816554296425*pi,1.656723536705833*pi) q[20];
U1q(0.345815443111086*pi,1.06339886355767*pi) q[21];
U1q(0.280601051027738*pi,0.0665081131203485*pi) q[22];
U1q(0.348737069649394*pi,1.107533891553926*pi) q[23];
RZZ(0.0*pi) q[0],q[6];
RZZ(0.0*pi) q[1],q[20];
RZZ(0.0*pi) q[2],q[23];
RZZ(0.0*pi) q[3],q[10];
RZZ(0.0*pi) q[4],q[16];
RZZ(0.0*pi) q[9],q[5];
RZZ(0.0*pi) q[11],q[7];
RZZ(0.0*pi) q[15],q[8];
RZZ(0.0*pi) q[14],q[12];
RZZ(0.0*pi) q[13],q[17];
RZZ(0.0*pi) q[21],q[18];
RZZ(0.0*pi) q[22],q[19];
rz(0.824480112028377*pi) q[0];
rz(3.311458506525669*pi) q[1];
rz(1.70581811317006*pi) q[2];
rz(3.159513293185916*pi) q[3];
rz(1.15893471144138*pi) q[4];
rz(1.11610098170528*pi) q[5];
rz(3.348614752850774*pi) q[6];
rz(2.4050593513919*pi) q[7];
rz(2.33776815293254*pi) q[8];
rz(0.0232256460914753*pi) q[9];
rz(0.705779329381674*pi) q[10];
rz(1.92859738635195*pi) q[11];
rz(0.334874715069767*pi) q[12];
rz(3.633924446404255*pi) q[13];
rz(3.800584410488137*pi) q[14];
rz(0.147142174402212*pi) q[15];
rz(0.638087553210934*pi) q[16];
rz(0.968688402799934*pi) q[17];
rz(2.13737710139645*pi) q[18];
rz(1.6193869735021*pi) q[19];
rz(3.617641153731805*pi) q[20];
rz(0.34334820976376*pi) q[21];
rz(1.24700028200185*pi) q[22];
rz(1.11719967005524*pi) q[23];
U1q(0.0685706042350537*pi,1.898948715871303*pi) q[0];
U1q(0.779583650280114*pi,1.820831929876979*pi) q[1];
U1q(0.604333846190585*pi,1.16662623821609*pi) q[2];
U1q(0.866559534356561*pi,1.712900611148322*pi) q[3];
U1q(0.48540651833713*pi,0.692368517425716*pi) q[4];
U1q(0.390419216720617*pi,0.723116879768224*pi) q[5];
U1q(0.516102214349984*pi,0.0572370041178556*pi) q[6];
U1q(0.669815978997818*pi,1.729455978459368*pi) q[7];
U1q(0.650031084764658*pi,1.062959325558809*pi) q[8];
U1q(0.321402984598494*pi,0.650466116900629*pi) q[9];
U1q(0.931994329441438*pi,0.208318188562655*pi) q[10];
U1q(0.643442250308782*pi,1.26945455528679*pi) q[11];
U1q(0.596006071244227*pi,1.921728942907215*pi) q[12];
U1q(0.507367369981718*pi,1.9077952853003206*pi) q[13];
U1q(0.762124487193158*pi,0.336168264682485*pi) q[14];
U1q(0.49719160312158*pi,0.583435506544431*pi) q[15];
U1q(0.769110834644175*pi,0.782201246624532*pi) q[16];
U1q(0.89382198891281*pi,0.664971664872038*pi) q[17];
U1q(0.793775180748448*pi,1.73289790458552*pi) q[18];
U1q(0.492215554728561*pi,1.32491721333494*pi) q[19];
U1q(0.625429705153012*pi,0.135861856703002*pi) q[20];
U1q(0.361355487538235*pi,1.5798661717187161*pi) q[21];
U1q(0.568902479850228*pi,0.606349004087591*pi) q[22];
U1q(0.515540234512087*pi,0.931781510252723*pi) q[23];
RZZ(0.0*pi) q[13],q[0];
RZZ(0.0*pi) q[3],q[1];
RZZ(0.0*pi) q[17],q[2];
RZZ(0.0*pi) q[4],q[9];
RZZ(0.0*pi) q[5],q[10];
RZZ(0.0*pi) q[14],q[6];
RZZ(0.0*pi) q[7],q[16];
RZZ(0.0*pi) q[8],q[22];
RZZ(0.0*pi) q[11],q[23];
RZZ(0.0*pi) q[19],q[12];
RZZ(0.0*pi) q[15],q[18];
RZZ(0.0*pi) q[21],q[20];
rz(0.531409313187372*pi) q[0];
rz(3.902926795612038*pi) q[1];
rz(0.784033391384393*pi) q[2];
rz(0.616193755692186*pi) q[3];
rz(1.37653860941407*pi) q[4];
rz(2.15608788704572*pi) q[5];
rz(3.887513381672728*pi) q[6];
rz(2.56709898721434*pi) q[7];
rz(1.40231147748809*pi) q[8];
rz(3.505491279292997*pi) q[9];
rz(2.32297220296446*pi) q[10];
rz(0.777178518003585*pi) q[11];
rz(1.56506499452427*pi) q[12];
rz(0.0134765801694232*pi) q[13];
rz(0.539902516147049*pi) q[14];
rz(0.765622064800453*pi) q[15];
rz(1.02959099906713*pi) q[16];
rz(2.94321492717174*pi) q[17];
rz(0.295658598269219*pi) q[18];
rz(0.183135825852248*pi) q[19];
rz(0.552451458104008*pi) q[20];
rz(3.528880252017319*pi) q[21];
rz(0.735798614049468*pi) q[22];
rz(0.891565379822101*pi) q[23];
U1q(0.736241967508297*pi,0.366434812772175*pi) q[0];
U1q(0.236456997737159*pi,1.5047076768452259*pi) q[1];
U1q(0.324364029647207*pi,0.477484520744613*pi) q[2];
U1q(0.807035636863241*pi,0.697609101562058*pi) q[3];
U1q(0.424518558815424*pi,0.648962021648747*pi) q[4];
U1q(0.544186210831252*pi,1.077965058088789*pi) q[5];
U1q(0.549246256309414*pi,1.195483858871955*pi) q[6];
U1q(0.789422594655105*pi,1.456055047273896*pi) q[7];
U1q(0.496383197402574*pi,1.29358808486992*pi) q[8];
U1q(0.199711485950289*pi,0.9572129270530301*pi) q[9];
U1q(0.64779875080651*pi,1.755104706234864*pi) q[10];
U1q(0.673045832833436*pi,0.303806304852909*pi) q[11];
U1q(0.47765476538268*pi,1.26133563357195*pi) q[12];
U1q(0.494795120188923*pi,0.148422458135667*pi) q[13];
U1q(0.188341077144094*pi,1.708203973796487*pi) q[14];
U1q(0.686478814911194*pi,0.990903586285933*pi) q[15];
U1q(0.738371547972539*pi,0.914051439397174*pi) q[16];
U1q(0.578152218793066*pi,1.620649579088549*pi) q[17];
U1q(0.20190799859753*pi,0.35606951493549*pi) q[18];
U1q(0.419968016332957*pi,0.379143930156285*pi) q[19];
U1q(0.512195453280356*pi,0.244238081095888*pi) q[20];
U1q(0.633129156608677*pi,0.334642535534063*pi) q[21];
U1q(0.131557498358503*pi,0.12400743625683*pi) q[22];
U1q(0.222691905408222*pi,0.323130233719621*pi) q[23];
RZZ(0.0*pi) q[0],q[1];
RZZ(0.0*pi) q[14],q[2];
RZZ(0.0*pi) q[3],q[19];
RZZ(0.0*pi) q[4],q[23];
RZZ(0.0*pi) q[5],q[8];
RZZ(0.0*pi) q[10],q[6];
RZZ(0.0*pi) q[21],q[7];
RZZ(0.0*pi) q[9],q[17];
RZZ(0.0*pi) q[11],q[16];
RZZ(0.0*pi) q[22],q[12];
RZZ(0.0*pi) q[13],q[18];
RZZ(0.0*pi) q[15],q[20];
rz(1.44461100188778*pi) q[0];
rz(0.62075433192784*pi) q[1];
rz(1.77341150867764*pi) q[2];
rz(1.4592441955117*pi) q[3];
rz(1.66217552566064*pi) q[4];
rz(2.61937857894572*pi) q[5];
rz(3.872199855460875*pi) q[6];
rz(2.4493087690496402*pi) q[7];
rz(1.09579084772545*pi) q[8];
rz(0.806872456953341*pi) q[9];
rz(0.194341663679507*pi) q[10];
rz(3.837116163269985*pi) q[11];
rz(0.845586510817853*pi) q[12];
rz(0.97564557374276*pi) q[13];
rz(0.0387290315011224*pi) q[14];
rz(3.770003012748108*pi) q[15];
rz(1.81242694770568*pi) q[16];
rz(0.348771069274736*pi) q[17];
rz(2.03847672160988*pi) q[18];
rz(3.741216128273858*pi) q[19];
rz(1.56376300890317*pi) q[20];
rz(0.151453467926874*pi) q[21];
rz(0.322534349801017*pi) q[22];
rz(3.349195983433318*pi) q[23];
U1q(0.384800778309221*pi,0.110574469570822*pi) q[0];
U1q(0.504476385238011*pi,0.339015368727725*pi) q[1];
U1q(0.706413965052135*pi,1.10400680279222*pi) q[2];
U1q(0.266943119053836*pi,1.9652407211829033*pi) q[3];
U1q(0.648139757827619*pi,1.41401856010336*pi) q[4];
U1q(0.578216885334327*pi,1.246920426284429*pi) q[5];
U1q(0.557379968568189*pi,0.732897406489088*pi) q[6];
U1q(0.646819905148279*pi,1.620932763019184*pi) q[7];
U1q(0.312779376054416*pi,1.697265644329892*pi) q[8];
U1q(0.211377264435075*pi,1.938027559709226*pi) q[9];
U1q(0.686268199462187*pi,1.9158444806910615*pi) q[10];
U1q(0.150252079598002*pi,0.606551066099571*pi) q[11];
U1q(0.635115139534638*pi,0.756333320016212*pi) q[12];
U1q(0.236111695064386*pi,1.14570421316536*pi) q[13];
U1q(0.158976285828736*pi,1.3798011278180229*pi) q[14];
U1q(0.410720513825789*pi,0.203502211773121*pi) q[15];
U1q(0.738293883412775*pi,1.50236547662466*pi) q[16];
U1q(0.10644411471394*pi,0.202586432686338*pi) q[17];
U1q(0.670444979925395*pi,0.8727243379224501*pi) q[18];
U1q(0.709688474518077*pi,1.709982402700236*pi) q[19];
U1q(0.509281616488825*pi,0.856268793459826*pi) q[20];
U1q(0.329754012431006*pi,1.689476458736731*pi) q[21];
U1q(0.425708860203948*pi,1.675180031507752*pi) q[22];
U1q(0.676648568706348*pi,0.258266538423642*pi) q[23];
RZZ(0.0*pi) q[0],q[11];
RZZ(0.0*pi) q[1],q[7];
RZZ(0.0*pi) q[4],q[2];
RZZ(0.0*pi) q[3],q[8];
RZZ(0.0*pi) q[21],q[5];
RZZ(0.0*pi) q[6],q[12];
RZZ(0.0*pi) q[9],q[18];
RZZ(0.0*pi) q[10],q[16];
RZZ(0.0*pi) q[13],q[23];
RZZ(0.0*pi) q[15],q[14];
RZZ(0.0*pi) q[17],q[22];
RZZ(0.0*pi) q[19],q[20];
rz(1.44863934186711*pi) q[0];
rz(3.9398260293765817*pi) q[1];
rz(0.172384698763752*pi) q[2];
rz(0.68291100019416*pi) q[3];
rz(0.839316944084667*pi) q[4];
rz(1.64493820331237*pi) q[5];
rz(0.581913699335755*pi) q[6];
rz(0.537387641456029*pi) q[7];
rz(0.88469870934448*pi) q[8];
rz(1.46640933072041*pi) q[9];
rz(3.828425963237928*pi) q[10];
rz(1.08052956934195*pi) q[11];
rz(0.726538543028209*pi) q[12];
rz(0.638660591311882*pi) q[13];
rz(3.431897172032662*pi) q[14];
rz(1.06153678435449*pi) q[15];
rz(0.211601021192197*pi) q[16];
rz(0.740780284445833*pi) q[17];
rz(1.49388225829791*pi) q[18];
rz(0.538340875623031*pi) q[19];
rz(3.725760970228469*pi) q[20];
rz(2.4990522771622903*pi) q[21];
rz(0.32462554638738*pi) q[22];
rz(3.907068229743568*pi) q[23];
U1q(0.665246163226316*pi,1.2155474153503*pi) q[0];
U1q(0.172642051758131*pi,1.107668993829519*pi) q[1];
U1q(0.555928858857399*pi,1.9724291508159333*pi) q[2];
U1q(0.544412228595448*pi,0.114967572396664*pi) q[3];
U1q(0.737948767284596*pi,0.829856440644792*pi) q[4];
U1q(0.518571760195986*pi,0.831010319066117*pi) q[5];
U1q(0.292373003057057*pi,1.657834356899359*pi) q[6];
U1q(0.372859693423367*pi,0.975686431178308*pi) q[7];
U1q(0.347444820697044*pi,1.597060604329916*pi) q[8];
U1q(0.383243634232008*pi,0.732904351691484*pi) q[9];
U1q(0.480977976503786*pi,0.875024328756244*pi) q[10];
U1q(0.0990569830517525*pi,0.46311915888097*pi) q[11];
U1q(0.60682745774723*pi,1.04503008277472*pi) q[12];
U1q(0.263967831111984*pi,0.549720624570738*pi) q[13];
U1q(0.801642129808031*pi,1.711917510240843*pi) q[14];
U1q(0.469750749369006*pi,0.498730516806064*pi) q[15];
U1q(0.200253909813779*pi,1.322145082823347*pi) q[16];
U1q(0.520836171218187*pi,1.683030269763294*pi) q[17];
U1q(0.206483603094041*pi,1.57683051156322*pi) q[18];
U1q(0.504644101601305*pi,0.250622746849063*pi) q[19];
U1q(0.421244986967401*pi,0.108421756331296*pi) q[20];
U1q(0.816728731022398*pi,1.042826797335139*pi) q[21];
U1q(0.102330467601714*pi,1.0633589011106*pi) q[22];
U1q(0.172544164804989*pi,1.882086959217465*pi) q[23];
RZZ(0.0*pi) q[8],q[0];
RZZ(0.0*pi) q[2],q[1];
RZZ(0.0*pi) q[15],q[3];
RZZ(0.0*pi) q[4],q[17];
RZZ(0.0*pi) q[5],q[7];
RZZ(0.0*pi) q[6],q[19];
RZZ(0.0*pi) q[9],q[23];
RZZ(0.0*pi) q[21],q[10];
RZZ(0.0*pi) q[11],q[12];
RZZ(0.0*pi) q[13],q[20];
RZZ(0.0*pi) q[14],q[22];
RZZ(0.0*pi) q[18],q[16];
rz(0.638180474352282*pi) q[0];
rz(2.78044780315782*pi) q[1];
rz(0.460917883336623*pi) q[2];
rz(3.426487078661267*pi) q[3];
rz(1.2710087492255*pi) q[4];
rz(2.94809733074434*pi) q[5];
rz(3.460059138926311*pi) q[6];
rz(3.551729885686237*pi) q[7];
rz(1.03548656975486*pi) q[8];
rz(2.0107841004236597*pi) q[9];
rz(0.64489103960487*pi) q[10];
rz(0.715557699349583*pi) q[11];
rz(3.093327410547624*pi) q[12];
rz(1.87365485544838*pi) q[13];
rz(0.589977174087346*pi) q[14];
rz(0.211299625387787*pi) q[15];
rz(0.279512625170902*pi) q[16];
rz(3.771141062806454*pi) q[17];
rz(0.720520104489836*pi) q[18];
rz(2.3225264779380197*pi) q[19];
rz(3.52303502542342*pi) q[20];
rz(3.31466005888301*pi) q[21];
rz(2.9980003755024*pi) q[22];
rz(1.01850026527419*pi) q[23];
U1q(0.284894353363024*pi,1.850552932286799*pi) q[0];
U1q(0.826696170754735*pi,1.9642153488599028*pi) q[1];
U1q(0.570837839159674*pi,0.60848345896433*pi) q[2];
U1q(0.933434796622657*pi,0.278953644519948*pi) q[3];
U1q(0.282933192042152*pi,0.496300768362887*pi) q[4];
U1q(0.790943895017431*pi,1.4922468490307002*pi) q[5];
U1q(0.493353020859231*pi,1.580297666921164*pi) q[6];
U1q(0.138807544671249*pi,1.878943137241438*pi) q[7];
U1q(0.589772626008519*pi,1.775536607545179*pi) q[8];
U1q(0.773192153951328*pi,1.742054913964872*pi) q[9];
U1q(0.120410832343349*pi,0.85607879114145*pi) q[10];
U1q(0.781877237752836*pi,0.876848628827632*pi) q[11];
U1q(0.864106244832137*pi,1.764778586627064*pi) q[12];
U1q(0.878610529023833*pi,1.33136199645638*pi) q[13];
U1q(0.884355208875429*pi,0.0589665477850074*pi) q[14];
U1q(0.188471938217164*pi,1.386420258422991*pi) q[15];
U1q(0.571415960054924*pi,0.273379108226942*pi) q[16];
U1q(0.884326048635608*pi,0.075236333128923*pi) q[17];
U1q(0.362067865492459*pi,0.398940502646647*pi) q[18];
U1q(0.519425255147617*pi,1.709587139594934*pi) q[19];
U1q(0.163569306885992*pi,1.784573331712496*pi) q[20];
U1q(0.707439920472998*pi,1.624412765827186*pi) q[21];
U1q(0.665184594983939*pi,1.833018541134361*pi) q[22];
U1q(0.379378792990954*pi,0.650182662436221*pi) q[23];
RZZ(0.0*pi) q[15],q[0];
RZZ(0.0*pi) q[4],q[1];
RZZ(0.0*pi) q[2],q[20];
RZZ(0.0*pi) q[3],q[16];
RZZ(0.0*pi) q[5],q[22];
RZZ(0.0*pi) q[17],q[6];
RZZ(0.0*pi) q[9],q[7];
RZZ(0.0*pi) q[13],q[8];
RZZ(0.0*pi) q[10],q[14];
RZZ(0.0*pi) q[11],q[19];
RZZ(0.0*pi) q[18],q[12];
RZZ(0.0*pi) q[21],q[23];
rz(0.794571939461699*pi) q[0];
rz(0.210256958922367*pi) q[1];
rz(3.198228544239597*pi) q[2];
rz(1.14157911991709*pi) q[3];
rz(3.9658903403898194*pi) q[4];
rz(0.981449540827298*pi) q[5];
rz(0.520037412643157*pi) q[6];
rz(2.97479509258179*pi) q[7];
rz(0.705054056880391*pi) q[8];
rz(0.21932067021503*pi) q[9];
rz(0.477380003289991*pi) q[10];
rz(0.0385377733299971*pi) q[11];
rz(1.05277993184346*pi) q[12];
rz(3.382313461960436*pi) q[13];
rz(3.60854618377056*pi) q[14];
rz(2.30892939832588*pi) q[15];
rz(1.10612232751494*pi) q[16];
rz(0.540802535899726*pi) q[17];
rz(1.04296171695534*pi) q[18];
rz(0.228355310191843*pi) q[19];
rz(3.619151717425579*pi) q[20];
rz(3.127359158970347*pi) q[21];
rz(0.394232795936438*pi) q[22];
rz(0.367703991555117*pi) q[23];
U1q(0.440818909980857*pi,1.9589188734996204*pi) q[0];
U1q(0.162412161349093*pi,0.225978094720703*pi) q[1];
U1q(0.74599503085427*pi,1.421300408583503*pi) q[2];
U1q(0.807878810500127*pi,1.30333635616008*pi) q[3];
U1q(0.236992270624375*pi,1.137167039600204*pi) q[4];
U1q(0.424845009144066*pi,0.174620037582132*pi) q[5];
U1q(0.125742240652747*pi,0.141287471918461*pi) q[6];
U1q(0.670032450387863*pi,1.675637609821784*pi) q[7];
U1q(0.296850784648888*pi,1.2498383543889*pi) q[8];
U1q(0.379026123557573*pi,0.799515034924768*pi) q[9];
U1q(0.777376259871316*pi,0.216539071920497*pi) q[10];
U1q(0.534147715580718*pi,0.285372871433219*pi) q[11];
U1q(0.297201915566576*pi,0.708593535546993*pi) q[12];
U1q(0.752320705478127*pi,0.351748864911367*pi) q[13];
U1q(0.655034839900752*pi,1.9804855635769907*pi) q[14];
U1q(0.645078161463869*pi,1.502006311980401*pi) q[15];
U1q(0.263188867539755*pi,1.11285581200503*pi) q[16];
U1q(0.399155637720673*pi,0.824672973060223*pi) q[17];
U1q(0.325244530398184*pi,0.626190055961223*pi) q[18];
U1q(0.267536841377546*pi,1.9717305662694005*pi) q[19];
U1q(0.398597472950095*pi,0.415383185490612*pi) q[20];
U1q(0.666595840988619*pi,1.447215576261911*pi) q[21];
U1q(0.491641043255686*pi,0.593128243850465*pi) q[22];
U1q(0.882239939102735*pi,0.0334076560018077*pi) q[23];
RZZ(0.0*pi) q[21],q[0];
RZZ(0.0*pi) q[15],q[1];
RZZ(0.0*pi) q[2],q[16];
RZZ(0.0*pi) q[3],q[12];
RZZ(0.0*pi) q[4],q[7];
RZZ(0.0*pi) q[5],q[20];
RZZ(0.0*pi) q[11],q[6];
RZZ(0.0*pi) q[8],q[14];
RZZ(0.0*pi) q[9],q[22];
RZZ(0.0*pi) q[13],q[10];
RZZ(0.0*pi) q[17],q[19];
RZZ(0.0*pi) q[18],q[23];
rz(1.62588235435604*pi) q[0];
rz(1.19880466295612*pi) q[1];
rz(0.0433910671705056*pi) q[2];
rz(3.9301497796476683*pi) q[3];
rz(0.421886294322449*pi) q[4];
rz(0.0293611495710796*pi) q[5];
rz(3.9013217919630296*pi) q[6];
rz(0.680457759937265*pi) q[7];
rz(1.03603710595276*pi) q[8];
rz(2.9933624747441403*pi) q[9];
rz(0.0727697633253652*pi) q[10];
rz(3.829011585420536*pi) q[11];
rz(1.04608463081404*pi) q[12];
rz(3.541676659598208*pi) q[13];
rz(0.476381864437512*pi) q[14];
rz(1.26116256320862*pi) q[15];
rz(0.776266569248698*pi) q[16];
rz(3.841686688491846*pi) q[17];
rz(0.168678863509569*pi) q[18];
rz(0.495220068465413*pi) q[19];
rz(3.864433453690309*pi) q[20];
rz(2.37375978049393*pi) q[21];
rz(1.33971349594511*pi) q[22];
rz(1.18156276277455*pi) q[23];
U1q(0.674235470733434*pi,0.970504082222432*pi) q[0];
U1q(0.207455418473205*pi,1.671716538878687*pi) q[1];
U1q(0.497531299016766*pi,0.852127201333618*pi) q[2];
U1q(0.404679959907129*pi,1.248334196818945*pi) q[3];
U1q(0.702627311559783*pi,0.374849145256139*pi) q[4];
U1q(0.621388659875489*pi,0.249162564330298*pi) q[5];
U1q(0.282742483125871*pi,1.338593796132597*pi) q[6];
U1q(0.626488321893308*pi,0.356254416829734*pi) q[7];
U1q(0.130617848106783*pi,0.294751911572588*pi) q[8];
U1q(0.58623659332942*pi,1.855662589625268*pi) q[9];
U1q(0.280268588177033*pi,1.292153372043684*pi) q[10];
U1q(0.502056892015811*pi,1.010645851430764*pi) q[11];
U1q(0.753893446720269*pi,0.678799004392753*pi) q[12];
U1q(0.33694363980177*pi,1.177378345910008*pi) q[13];
U1q(0.359858049999567*pi,1.628684369105934*pi) q[14];
U1q(0.292211708919917*pi,0.0742838647757098*pi) q[15];
U1q(0.515228550458973*pi,0.585324029930746*pi) q[16];
U1q(0.802859168394094*pi,0.434622369530466*pi) q[17];
U1q(0.3561563844598*pi,0.877940775939857*pi) q[18];
U1q(0.474853262189774*pi,0.799532201158008*pi) q[19];
U1q(0.470453052574607*pi,0.510380207425438*pi) q[20];
U1q(0.911958503294594*pi,1.358767744473748*pi) q[21];
U1q(0.134098709398572*pi,1.13823408118256*pi) q[22];
U1q(0.658033811526806*pi,1.10922867482516*pi) q[23];
RZZ(0.0*pi) q[9],q[0];
RZZ(0.0*pi) q[1],q[16];
RZZ(0.0*pi) q[11],q[2];
RZZ(0.0*pi) q[3],q[13];
RZZ(0.0*pi) q[21],q[4];
RZZ(0.0*pi) q[5],q[12];
RZZ(0.0*pi) q[15],q[6];
RZZ(0.0*pi) q[7],q[19];
RZZ(0.0*pi) q[8],q[17];
RZZ(0.0*pi) q[10],q[18];
RZZ(0.0*pi) q[14],q[20];
RZZ(0.0*pi) q[22],q[23];
rz(0.192343079728634*pi) q[0];
rz(0.98696375990946*pi) q[1];
rz(1.40486622467258*pi) q[2];
rz(0.393080205063062*pi) q[3];
rz(0.322997407444072*pi) q[4];
rz(1.44384787352442*pi) q[5];
rz(1.98657405743264*pi) q[6];
rz(0.756759468983967*pi) q[7];
rz(2.27819965764927*pi) q[8];
rz(1.489036249206*pi) q[9];
rz(1.46336477024895*pi) q[10];
rz(2.2845125996120803*pi) q[11];
rz(1.65492677070299*pi) q[12];
rz(0.618645976888104*pi) q[13];
rz(3.864053190256355*pi) q[14];
rz(3.759574188650602*pi) q[15];
rz(2.71427920318151*pi) q[16];
rz(0.806810420912753*pi) q[17];
rz(0.448567976134846*pi) q[18];
rz(1.2528071120565*pi) q[19];
rz(3.504099140386656*pi) q[20];
rz(1.86081454077967*pi) q[21];
rz(0.0764767025382446*pi) q[22];
rz(0.0222972694779048*pi) q[23];
U1q(0.385775428219192*pi,1.67636838682963*pi) q[0];
U1q(0.532094472950218*pi,1.17970430285935*pi) q[1];
U1q(0.906927210848716*pi,0.696575073334669*pi) q[2];
U1q(0.548935192203601*pi,0.295481066960411*pi) q[3];
U1q(0.582265649847396*pi,0.672048150904883*pi) q[4];
U1q(0.324906666943608*pi,1.845699008124672*pi) q[5];
U1q(0.775491702846442*pi,1.22575507238399*pi) q[6];
U1q(0.630200508326341*pi,0.299210440821628*pi) q[7];
U1q(0.66412903230962*pi,1.623941880686208*pi) q[8];
U1q(0.198300793530133*pi,1.47136309491913*pi) q[9];
U1q(0.318903485353526*pi,0.157079791604915*pi) q[10];
U1q(0.699992885955095*pi,1.2716805496961792*pi) q[11];
U1q(0.768505073811693*pi,1.50408143142508*pi) q[12];
U1q(0.161849995065969*pi,0.518258553214374*pi) q[13];
U1q(0.408557948550516*pi,0.853481900478595*pi) q[14];
U1q(0.748907983405795*pi,1.788439475610069*pi) q[15];
U1q(0.661743322257974*pi,1.865360226146513*pi) q[16];
U1q(0.227597565406122*pi,1.37126687659265*pi) q[17];
U1q(0.16506400229603*pi,0.0921786653175268*pi) q[18];
U1q(0.146834030026962*pi,1.0930368350908*pi) q[19];
U1q(0.529323525542346*pi,1.6584076359460331*pi) q[20];
U1q(0.698633756833709*pi,1.04063825146416*pi) q[21];
U1q(0.711162575602804*pi,1.9994884313695784*pi) q[22];
U1q(0.369833571206483*pi,0.885971086794963*pi) q[23];
RZZ(0.0*pi) q[0],q[20];
RZZ(0.0*pi) q[1],q[19];
RZZ(0.0*pi) q[5],q[2];
RZZ(0.0*pi) q[3],q[6];
RZZ(0.0*pi) q[4],q[8];
RZZ(0.0*pi) q[10],q[7];
RZZ(0.0*pi) q[9],q[16];
RZZ(0.0*pi) q[13],q[11];
RZZ(0.0*pi) q[21],q[12];
RZZ(0.0*pi) q[14],q[23];
RZZ(0.0*pi) q[15],q[17];
RZZ(0.0*pi) q[18],q[22];
rz(2.57385702314805*pi) q[0];
rz(1.08159218214289*pi) q[1];
rz(0.479287337906992*pi) q[2];
rz(1.06436329034575*pi) q[3];
rz(3.419031525704204*pi) q[4];
rz(0.337267685264781*pi) q[5];
rz(3.847259867827527*pi) q[6];
rz(1.73242589766008*pi) q[7];
rz(0.998561511020055*pi) q[8];
rz(3.6536805353421338*pi) q[9];
rz(0.745166475044356*pi) q[10];
rz(1.81787947850004*pi) q[11];
rz(0.604922785290301*pi) q[12];
rz(2.63076206997241*pi) q[13];
rz(1.0344889939173*pi) q[14];
rz(1.2974770678086*pi) q[15];
rz(1.9803025566947*pi) q[16];
rz(3.399312235661236*pi) q[17];
rz(1.96139319185864*pi) q[18];
rz(3.90672650255099*pi) q[19];
rz(1.40305304437169*pi) q[20];
rz(3.949146838230828*pi) q[21];
rz(2.99591338952929*pi) q[22];
rz(0.438188196813793*pi) q[23];
U1q(0.780543325599351*pi,1.632623544042167*pi) q[0];
U1q(0.179147494838568*pi,1.882443724240977*pi) q[1];
U1q(0.817540096508095*pi,0.918615246122913*pi) q[2];
U1q(0.447158834818472*pi,1.50781610705333*pi) q[3];
U1q(0.481919641923266*pi,1.870053457388042*pi) q[4];
U1q(0.431237829439419*pi,1.878817651873175*pi) q[5];
U1q(0.269031128916478*pi,1.749637979615023*pi) q[6];
U1q(0.520129986499552*pi,0.811677641201612*pi) q[7];
U1q(0.249076089884968*pi,0.46770998559995*pi) q[8];
U1q(0.32852604672217*pi,1.9289517947620944*pi) q[9];
U1q(0.898502216759714*pi,0.431713955976905*pi) q[10];
U1q(0.598106014728679*pi,1.14133103518594*pi) q[11];
U1q(0.233735372993174*pi,1.9213778395030787*pi) q[12];
U1q(0.634664297882482*pi,0.0572582190518829*pi) q[13];
U1q(0.54929289249641*pi,0.327948452585208*pi) q[14];
U1q(0.440080021125901*pi,0.651900887892704*pi) q[15];
U1q(0.591325575860804*pi,0.929277493867816*pi) q[16];
U1q(0.451946100241082*pi,1.770087413382768*pi) q[17];
U1q(0.75612939510334*pi,1.54477470182723*pi) q[18];
U1q(0.438551576369325*pi,1.310497146366475*pi) q[19];
U1q(0.246908751922971*pi,1.8610301808842*pi) q[20];
U1q(0.129439581971645*pi,0.73256083122415*pi) q[21];
U1q(0.905252249446856*pi,1.860695839904209*pi) q[22];
U1q(0.526982114019296*pi,0.0115613066326885*pi) q[23];
RZZ(0.0*pi) q[0],q[12];
RZZ(0.0*pi) q[14],q[1];
RZZ(0.0*pi) q[9],q[2];
RZZ(0.0*pi) q[3],q[5];
RZZ(0.0*pi) q[4],q[20];
RZZ(0.0*pi) q[6],q[16];
RZZ(0.0*pi) q[23],q[7];
RZZ(0.0*pi) q[10],q[8];
RZZ(0.0*pi) q[17],q[11];
RZZ(0.0*pi) q[15],q[13];
RZZ(0.0*pi) q[18],q[19];
RZZ(0.0*pi) q[21],q[22];
rz(1.07937291185566*pi) q[0];
rz(0.394337430633165*pi) q[1];
rz(0.126876950162188*pi) q[2];
rz(0.903862105144265*pi) q[3];
rz(1.61798841184871*pi) q[4];
rz(3.9570479646730408*pi) q[5];
rz(0.416458115977308*pi) q[6];
rz(3.505612014719361*pi) q[7];
rz(0.628155351570496*pi) q[8];
rz(3.893926669898992*pi) q[9];
rz(3.596667994816502*pi) q[10];
rz(0.428755956274167*pi) q[11];
rz(3.9158839095918796*pi) q[12];
rz(2.80686280814712*pi) q[13];
rz(3.385994991727098*pi) q[14];
rz(0.716549457094994*pi) q[15];
rz(0.185666599148146*pi) q[16];
rz(1.97352867285603*pi) q[17];
rz(3.621406551697044*pi) q[18];
rz(1.18018668081233*pi) q[19];
rz(3.837342937440165*pi) q[20];
rz(3.899883742701636*pi) q[21];
rz(3.7003052454889422*pi) q[22];
rz(3.4696571252972*pi) q[23];
U1q(0.83917099214963*pi,0.558194511995485*pi) q[0];
U1q(0.171876581717206*pi,1.9207374721364505*pi) q[1];
U1q(0.872389997125018*pi,0.292400625979841*pi) q[2];
U1q(0.415679462479839*pi,1.752732920423213*pi) q[3];
U1q(0.664162586098866*pi,0.966526008026038*pi) q[4];
U1q(0.641031956303919*pi,0.189809688320606*pi) q[5];
U1q(0.267278988625481*pi,1.7643831473512561*pi) q[6];
U1q(0.287342126037317*pi,0.8418431981346901*pi) q[7];
U1q(0.370360465709363*pi,0.554422083327061*pi) q[8];
U1q(0.306221224373301*pi,1.517785879319325*pi) q[9];
U1q(0.0739019497916947*pi,1.703965257958088*pi) q[10];
U1q(0.510121514122319*pi,0.277451864537138*pi) q[11];
U1q(0.438939679207911*pi,0.590979062046744*pi) q[12];
U1q(0.793962304236715*pi,0.143552300107672*pi) q[13];
U1q(0.890785540796619*pi,0.085618446041715*pi) q[14];
U1q(0.260743110170534*pi,0.0552572769064019*pi) q[15];
U1q(0.526989377390495*pi,0.259296063569036*pi) q[16];
U1q(0.640811107368425*pi,1.46920563599602*pi) q[17];
U1q(0.299272518648025*pi,0.190128695883965*pi) q[18];
U1q(0.267352952993572*pi,1.45062241869616*pi) q[19];
U1q(0.318836396055636*pi,0.814974627091381*pi) q[20];
U1q(0.667551247524106*pi,1.736423638472479*pi) q[21];
U1q(0.359761480159014*pi,1.795637766045005*pi) q[22];
U1q(0.597894366407245*pi,1.9260945164222458*pi) q[23];
RZZ(0.0*pi) q[0],q[17];
RZZ(0.0*pi) q[10],q[1];
RZZ(0.0*pi) q[13],q[2];
RZZ(0.0*pi) q[3],q[23];
RZZ(0.0*pi) q[4],q[12];
RZZ(0.0*pi) q[5],q[6];
RZZ(0.0*pi) q[18],q[7];
RZZ(0.0*pi) q[9],q[8];
RZZ(0.0*pi) q[14],q[11];
RZZ(0.0*pi) q[15],q[21];
RZZ(0.0*pi) q[19],q[16];
RZZ(0.0*pi) q[22],q[20];
rz(0.389861881662853*pi) q[0];
rz(0.501596848793046*pi) q[1];
rz(0.351324153145755*pi) q[2];
rz(3.897945022629989*pi) q[3];
rz(3.898932247971387*pi) q[4];
rz(3.89999377207739*pi) q[5];
rz(0.619166824355617*pi) q[6];
rz(3.741574540741043*pi) q[7];
rz(3.477419925768355*pi) q[8];
rz(1.61945638408292*pi) q[9];
rz(3.273176160575857*pi) q[10];
rz(0.193980004781011*pi) q[11];
rz(3.670469471930245*pi) q[12];
rz(0.926803849819629*pi) q[13];
rz(1.52024462148696*pi) q[14];
rz(1.81196133953947*pi) q[15];
rz(1.57901907370259*pi) q[16];
rz(1.59600594251769*pi) q[17];
rz(1.2401722019332*pi) q[18];
rz(0.460376444290233*pi) q[19];
rz(1.27612487153096*pi) q[20];
rz(1.54806568235227*pi) q[21];
rz(0.441051859966166*pi) q[22];
rz(3.313386715527996*pi) q[23];
U1q(0.850149621259855*pi,8.9850528340818e-05*pi) q[0];
U1q(0.615628976983111*pi,0.55550543864161*pi) q[1];
U1q(0.120325613223199*pi,0.334672396501211*pi) q[2];
U1q(0.313819979185207*pi,1.317160511163725*pi) q[3];
U1q(0.367390721233711*pi,1.068050497157095*pi) q[4];
U1q(0.595147252513417*pi,1.7286477616429279*pi) q[5];
U1q(0.304096334261793*pi,0.169554740032567*pi) q[6];
U1q(0.288219011555468*pi,1.002285520287157*pi) q[7];
U1q(0.570490805351625*pi,0.387710218832255*pi) q[8];
U1q(0.658969314927312*pi,0.846864165507417*pi) q[9];
U1q(0.543025510366175*pi,1.495210980621768*pi) q[10];
U1q(0.798562034183593*pi,0.216846479943486*pi) q[11];
U1q(0.916088558481866*pi,0.337115990124151*pi) q[12];
U1q(0.373767560117301*pi,1.88538373348551*pi) q[13];
U1q(0.487378307312732*pi,1.20632767369871*pi) q[14];
U1q(0.795031313526273*pi,0.701043345452329*pi) q[15];
U1q(0.576848055828224*pi,1.39273015008684*pi) q[16];
U1q(0.469582691857106*pi,1.2352531702776*pi) q[17];
U1q(0.635803881971552*pi,0.985596903487565*pi) q[18];
U1q(0.276279484658743*pi,1.678796336239043*pi) q[19];
U1q(0.353279274691023*pi,1.76347753142093*pi) q[20];
U1q(0.427216950963205*pi,1.21106392651683*pi) q[21];
U1q(0.276923007801375*pi,1.4054384033928708*pi) q[22];
U1q(0.775331195562127*pi,1.806511145884301*pi) q[23];
RZZ(0.0*pi) q[0],q[14];
RZZ(0.0*pi) q[22],q[1];
RZZ(0.0*pi) q[15],q[2];
RZZ(0.0*pi) q[3],q[17];
RZZ(0.0*pi) q[4],q[10];
RZZ(0.0*pi) q[5],q[11];
RZZ(0.0*pi) q[9],q[6];
RZZ(0.0*pi) q[13],q[7];
RZZ(0.0*pi) q[8],q[18];
RZZ(0.0*pi) q[12],q[16];
RZZ(0.0*pi) q[21],q[19];
RZZ(0.0*pi) q[23],q[20];
rz(2.77678232493826*pi) q[0];
rz(0.571070449649428*pi) q[1];
rz(1.0265257728831*pi) q[2];
rz(0.489502477678323*pi) q[3];
rz(0.55046838491574*pi) q[4];
rz(0.351344271065149*pi) q[5];
rz(3.823766425918095*pi) q[6];
rz(0.0326708696542557*pi) q[7];
rz(0.676721521243352*pi) q[8];
rz(2.92108507766672*pi) q[9];
rz(3.034312575104344*pi) q[10];
rz(2.45277823659116*pi) q[11];
rz(1.89166588373459*pi) q[12];
rz(0.184242349136515*pi) q[13];
rz(0.35034033393723*pi) q[14];
rz(0.412208591108375*pi) q[15];
rz(1.07714925124931*pi) q[16];
rz(1.01401989722001*pi) q[17];
rz(3.274888797508979*pi) q[18];
rz(2.4193801181989203*pi) q[19];
rz(2.3100307715072903*pi) q[20];
rz(3.975472466428216*pi) q[21];
rz(3.944997834397931*pi) q[22];
rz(0.063825243308522*pi) q[23];
U1q(0.647285521949619*pi,1.382080605972202*pi) q[0];
U1q(0.638021971077273*pi,0.723765825942992*pi) q[1];
U1q(0.368762754568323*pi,1.41476454086712*pi) q[2];
U1q(0.74479488521818*pi,0.246130060235882*pi) q[3];
U1q(0.333493966066043*pi,0.128211133564368*pi) q[4];
U1q(0.263387031233394*pi,1.1391248287184*pi) q[5];
U1q(0.228946434856935*pi,1.872563742922299*pi) q[6];
U1q(0.724833957900861*pi,0.495460896622916*pi) q[7];
U1q(0.451810674034795*pi,1.615616874011913*pi) q[8];
U1q(0.596683749824274*pi,1.855418506279439*pi) q[9];
U1q(0.734498755448239*pi,1.9858547823657424*pi) q[10];
U1q(0.532115723238081*pi,1.774524728436107*pi) q[11];
U1q(0.565480283133702*pi,1.59340686117222*pi) q[12];
U1q(0.333616169829863*pi,1.646863671268963*pi) q[13];
U1q(0.440295435063832*pi,0.844851555102788*pi) q[14];
U1q(0.0873354974628792*pi,1.4784553877570081*pi) q[15];
U1q(0.609987822359095*pi,0.486595856475849*pi) q[16];
U1q(0.193014299613182*pi,1.577606475923887*pi) q[17];
U1q(0.71423916333587*pi,0.0827244991000931*pi) q[18];
U1q(0.888866484647371*pi,1.33796123361141*pi) q[19];
U1q(0.501125577091531*pi,1.748141206376202*pi) q[20];
U1q(0.378506775485333*pi,1.786765417351249*pi) q[21];
U1q(0.284835070645544*pi,0.838108680153017*pi) q[22];
U1q(0.374016932866773*pi,0.410760173818655*pi) q[23];
RZZ(0.0*pi) q[5],q[0];
RZZ(0.0*pi) q[17],q[1];
RZZ(0.0*pi) q[18],q[2];
RZZ(0.0*pi) q[4],q[3];
RZZ(0.0*pi) q[13],q[6];
RZZ(0.0*pi) q[15],q[7];
RZZ(0.0*pi) q[8],q[23];
RZZ(0.0*pi) q[9],q[20];
RZZ(0.0*pi) q[10],q[12];
RZZ(0.0*pi) q[22],q[11];
RZZ(0.0*pi) q[14],q[19];
RZZ(0.0*pi) q[21],q[16];
rz(2.08842288238424*pi) q[0];
rz(0.685804647706783*pi) q[1];
rz(1.17881902613212*pi) q[2];
rz(1.32743126255678*pi) q[3];
rz(0.584496390041009*pi) q[4];
rz(1.96749218132453*pi) q[5];
rz(3.503935423066293*pi) q[6];
rz(0.288760992317362*pi) q[7];
rz(0.282754173855747*pi) q[8];
rz(2.8858304366128698*pi) q[9];
rz(0.755156173561085*pi) q[10];
rz(0.0930548133420444*pi) q[11];
rz(3.682589223584213*pi) q[12];
rz(1.30925097970605*pi) q[13];
rz(0.00311994319677811*pi) q[14];
rz(3.9993318483259324*pi) q[15];
rz(0.424770484158518*pi) q[16];
rz(0.866266682041618*pi) q[17];
rz(1.06490852790914*pi) q[18];
rz(0.997277200288332*pi) q[19];
rz(0.907840667647779*pi) q[20];
rz(2.85218154397484*pi) q[21];
rz(2.13317546860207*pi) q[22];
rz(3.187403651488021*pi) q[23];
U1q(0.825084213007275*pi,1.072726181259978*pi) q[0];
U1q(0.708316608674238*pi,0.720230165331376*pi) q[1];
U1q(0.386927949630375*pi,0.0343360888141612*pi) q[2];
U1q(0.297193100824892*pi,1.9327993648592203*pi) q[3];
U1q(0.427318780155018*pi,1.552279273033969*pi) q[4];
U1q(0.600560641496028*pi,1.21948577791152*pi) q[5];
U1q(0.380696213022805*pi,1.600298918786162*pi) q[6];
U1q(0.030572633491656*pi,1.250885663621193*pi) q[7];
U1q(0.426926292216848*pi,0.944146120694433*pi) q[8];
U1q(0.687343305018418*pi,1.786712485397942*pi) q[9];
U1q(0.495758857912253*pi,0.938694484248928*pi) q[10];
U1q(0.657621910460473*pi,0.630514959283859*pi) q[11];
U1q(0.212011260477358*pi,0.290603630496891*pi) q[12];
U1q(0.524282226926663*pi,1.22867332832152*pi) q[13];
U1q(0.667372161296548*pi,1.9387320990531443*pi) q[14];
U1q(0.631066530796857*pi,0.197228812754536*pi) q[15];
U1q(0.184244345968272*pi,0.882880914918187*pi) q[16];
U1q(0.734125392209297*pi,0.705669595337131*pi) q[17];
U1q(0.897546401779369*pi,0.346067130904845*pi) q[18];
U1q(0.225292519174796*pi,1.37316784614382*pi) q[19];
U1q(0.218928641661622*pi,0.372699107690583*pi) q[20];
U1q(0.775957267745153*pi,1.770091157800341*pi) q[21];
U1q(0.561703719918111*pi,1.51496513313316*pi) q[22];
U1q(0.596164062824335*pi,1.889758836542704*pi) q[23];
RZZ(0.0*pi) q[0],q[16];
RZZ(0.0*pi) q[13],q[1];
RZZ(0.0*pi) q[2],q[6];
RZZ(0.0*pi) q[3],q[11];
RZZ(0.0*pi) q[4],q[22];
RZZ(0.0*pi) q[15],q[5];
RZZ(0.0*pi) q[7],q[20];
RZZ(0.0*pi) q[8],q[12];
RZZ(0.0*pi) q[9],q[19];
RZZ(0.0*pi) q[10],q[23];
RZZ(0.0*pi) q[18],q[14];
RZZ(0.0*pi) q[21],q[17];
rz(0.135001461367655*pi) q[0];
rz(1.82391276506159*pi) q[1];
rz(1.72779577015991*pi) q[2];
rz(3.4202285580685308*pi) q[3];
rz(0.94393068497354*pi) q[4];
rz(1.7527926080944298*pi) q[5];
rz(3.833369748841934*pi) q[6];
rz(0.896876425279985*pi) q[7];
rz(3.778369922673009*pi) q[8];
rz(1.1258809843094002*pi) q[9];
rz(0.38436844188426*pi) q[10];
rz(2.86166795420151*pi) q[11];
rz(0.696105009953627*pi) q[12];
rz(0.07712510820246*pi) q[13];
rz(1.05663280918428*pi) q[14];
rz(1.1032762744287399*pi) q[15];
rz(1.28443610492837*pi) q[16];
rz(0.690148018548197*pi) q[17];
rz(0.421174284444502*pi) q[18];
rz(3.808589206365504*pi) q[19];
rz(1.52763063382752*pi) q[20];
rz(3.9655600070075097*pi) q[21];
rz(3.792407180028211*pi) q[22];
rz(1.14882050682599*pi) q[23];
U1q(3.206987523943044*pi,1.89066705417456*pi) q[0];
U1q(3.598291934712213*pi,1.70896736943224*pi) q[1];
U1q(3.718711108612996*pi,1.25761692985122*pi) q[2];
U1q(3.697849619780198*pi,1.9894817151614945*pi) q[3];
U1q(3.700011277368375*pi,0.172503670060759*pi) q[4];
U1q(3.640116472676871*pi,1.080731876034006*pi) q[5];
U1q(3.430460412208706*pi,1.345344443844334*pi) q[6];
U1q(3.48386351473207*pi,0.0693429318614054*pi) q[7];
U1q(3.653963845379966*pi,0.61565377896959*pi) q[8];
U1q(3.799619419990543*pi,1.570552638073059*pi) q[9];
U1q(3.473736635120149*pi,1.12964373095405*pi) q[10];
U1q(3.478374068439862*pi,1.9841833342465154*pi) q[11];
U1q(3.9387836957870745*pi,0.324096807205148*pi) q[12];
U1q(3.716647474573643*pi,1.813388999121999*pi) q[13];
U1q(3.326530716433797*pi,0.14077880667779*pi) q[14];
U1q(3.295749776916765*pi,1.255062148457571*pi) q[15];
U1q(3.388412854979657*pi,0.665266541819079*pi) q[16];
U1q(3.593953471641293*pi,1.9993039568901567*pi) q[17];
U1q(3.432538209853853*pi,0.0576489547014272*pi) q[18];
U1q(3.297956743579559*pi,0.715629733690205*pi) q[19];
U1q(3.335435395501179*pi,1.45529609741355*pi) q[20];
U1q(3.238918183718825*pi,1.687183011469164*pi) q[21];
U1q(3.613904739466579*pi,1.10080369189014*pi) q[22];
U1q(3.190488293888627*pi,0.665284755396522*pi) q[23];
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