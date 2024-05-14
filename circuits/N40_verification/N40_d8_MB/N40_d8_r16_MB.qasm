OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(0.718010875983119*pi,1.43106870278361*pi) q[0];
U1q(1.24790289234161*pi,0.9342791294271748*pi) q[1];
U1q(0.0838540770340036*pi,1.670008159506986*pi) q[2];
U1q(1.70042589573742*pi,1.2739126504174383*pi) q[3];
U1q(0.262700669926716*pi,0.318730791134039*pi) q[4];
U1q(0.531416851002644*pi,1.837502074540138*pi) q[5];
U1q(1.33331081149553*pi,0.7454422026778541*pi) q[6];
U1q(0.72777515337944*pi,0.0698012991983467*pi) q[7];
U1q(0.0171515746048202*pi,1.5628673902891261*pi) q[8];
U1q(0.545974558336321*pi,1.06004051256573*pi) q[9];
U1q(0.306865263437787*pi,0.824140334071252*pi) q[10];
U1q(0.65660443924699*pi,1.782935133255053*pi) q[11];
U1q(1.40253335277334*pi,0.58405250432436*pi) q[12];
U1q(0.542922923796314*pi,0.265218274746177*pi) q[13];
U1q(0.371207678667034*pi,1.528564354142171*pi) q[14];
U1q(1.27339818975086*pi,1.6890741904010713*pi) q[15];
U1q(1.49802151248513*pi,1.0707223801507602*pi) q[16];
U1q(0.825421834392657*pi,0.0429552191836893*pi) q[17];
U1q(1.32770096947093*pi,1.5589098172117102*pi) q[18];
U1q(1.76189907760846*pi,0.26886416571406874*pi) q[19];
U1q(0.450546942648371*pi,0.550421908788848*pi) q[20];
U1q(0.274176072840521*pi,0.0172716397578326*pi) q[21];
U1q(0.895249285060496*pi,0.901005463059984*pi) q[22];
U1q(0.241445008826735*pi,0.0175015825579246*pi) q[23];
U1q(1.18030295417209*pi,0.840591275759415*pi) q[24];
U1q(0.244263906710285*pi,0.215891663802341*pi) q[25];
U1q(1.37671462786717*pi,0.9147143134595457*pi) q[26];
U1q(0.532491925067819*pi,1.04534604502168*pi) q[27];
U1q(0.61214625516883*pi,0.31836892464744*pi) q[28];
U1q(1.5852480211268*pi,0.3229507516516164*pi) q[29];
U1q(3.381881296612801*pi,0.8289522807588511*pi) q[30];
U1q(0.66317332742233*pi,1.1734074004457868*pi) q[31];
U1q(1.75674276817185*pi,1.0623806978000534*pi) q[32];
U1q(1.30756349548356*pi,0.05617573831896816*pi) q[33];
U1q(0.685512688850475*pi,1.8803236285616*pi) q[34];
U1q(0.881055055383512*pi,0.765632440280573*pi) q[35];
U1q(0.330227396893231*pi,0.7043715671977999*pi) q[36];
U1q(1.41161008189456*pi,0.6966957808292448*pi) q[37];
U1q(1.10568776609996*pi,1.906820832369136*pi) q[38];
U1q(0.466077170324656*pi,0.563091605124215*pi) q[39];
RZZ(0.5*pi) q[11],q[0];
RZZ(0.5*pi) q[3],q[1];
RZZ(0.5*pi) q[31],q[2];
RZZ(0.5*pi) q[24],q[4];
RZZ(0.5*pi) q[5],q[29];
RZZ(0.5*pi) q[36],q[6];
RZZ(0.5*pi) q[7],q[17];
RZZ(0.5*pi) q[27],q[8];
RZZ(0.5*pi) q[15],q[9];
RZZ(0.5*pi) q[13],q[10];
RZZ(0.5*pi) q[12],q[35];
RZZ(0.5*pi) q[14],q[20];
RZZ(0.5*pi) q[16],q[21];
RZZ(0.5*pi) q[23],q[18];
RZZ(0.5*pi) q[39],q[19];
RZZ(0.5*pi) q[22],q[33];
RZZ(0.5*pi) q[25],q[34];
RZZ(0.5*pi) q[26],q[30];
RZZ(0.5*pi) q[37],q[28];
RZZ(0.5*pi) q[38],q[32];
U1q(0.0478054277901092*pi,1.611836447514395*pi) q[0];
U1q(0.624742535577449*pi,1.8069227610767749*pi) q[1];
U1q(0.463134251562901*pi,1.28704574885246*pi) q[2];
U1q(0.407552326551053*pi,0.6472064693243884*pi) q[3];
U1q(0.557685091153859*pi,0.05476121854027993*pi) q[4];
U1q(0.773087838449528*pi,1.4162551806413801*pi) q[5];
U1q(0.461020751208998*pi,1.8085464901725992*pi) q[6];
U1q(0.632671190613324*pi,1.3758695061980202*pi) q[7];
U1q(0.368860538755499*pi,0.6587895515902198*pi) q[8];
U1q(0.589242322514029*pi,1.392084506907059*pi) q[9];
U1q(0.201441349352272*pi,0.3611699535728401*pi) q[10];
U1q(0.213991364911671*pi,0.037978091438379824*pi) q[11];
U1q(0.397772109120771*pi,1.5623375866452198*pi) q[12];
U1q(0.795684049244395*pi,0.58691124101689*pi) q[13];
U1q(0.200816925353609*pi,1.71850728772164*pi) q[14];
U1q(0.299490389701247*pi,0.3335015066414009*pi) q[15];
U1q(0.573384261586947*pi,0.16867017504693327*pi) q[16];
U1q(0.40798292914069*pi,0.8713239484041999*pi) q[17];
U1q(0.578622103627882*pi,0.6235372650110582*pi) q[18];
U1q(0.801503294409482*pi,1.3548139586198478*pi) q[19];
U1q(0.73169760162945*pi,0.65188797001022*pi) q[20];
U1q(0.204539401539129*pi,0.09942383019861989*pi) q[21];
U1q(0.276278931207435*pi,1.349330881848783*pi) q[22];
U1q(0.639638613449092*pi,1.8656727615798698*pi) q[23];
U1q(0.58821160495107*pi,0.995529894753296*pi) q[24];
U1q(0.751832998165425*pi,0.17552672700957994*pi) q[25];
U1q(0.799687713319521*pi,1.1857558298686957*pi) q[26];
U1q(0.489191018509354*pi,0.652733423292025*pi) q[27];
U1q(0.699728128336986*pi,0.34580193319307995*pi) q[28];
U1q(0.681474948649817*pi,0.8463477093195064*pi) q[29];
U1q(0.20630984294307*pi,0.664820489695761*pi) q[30];
U1q(0.869820594767877*pi,1.6093333653697002*pi) q[31];
U1q(0.823230224403541*pi,1.4794617728676633*pi) q[32];
U1q(0.37670121296198*pi,1.0291068257066684*pi) q[33];
U1q(0.893408079011916*pi,0.3207678090822199*pi) q[34];
U1q(0.484629817125044*pi,0.6476243589722399*pi) q[35];
U1q(0.835233607510314*pi,1.99951314419528*pi) q[36];
U1q(0.322190053154461*pi,0.7261918834996548*pi) q[37];
U1q(0.291657888597398*pi,1.096477773076436*pi) q[38];
U1q(0.687838731696297*pi,1.9108422839074999*pi) q[39];
RZZ(0.5*pi) q[26],q[0];
RZZ(0.5*pi) q[25],q[1];
RZZ(0.5*pi) q[18],q[2];
RZZ(0.5*pi) q[3],q[29];
RZZ(0.5*pi) q[7],q[4];
RZZ(0.5*pi) q[38],q[5];
RZZ(0.5*pi) q[14],q[6];
RZZ(0.5*pi) q[30],q[8];
RZZ(0.5*pi) q[13],q[9];
RZZ(0.5*pi) q[15],q[10];
RZZ(0.5*pi) q[17],q[11];
RZZ(0.5*pi) q[21],q[12];
RZZ(0.5*pi) q[16],q[22];
RZZ(0.5*pi) q[19],q[35];
RZZ(0.5*pi) q[37],q[20];
RZZ(0.5*pi) q[23],q[32];
RZZ(0.5*pi) q[36],q[24];
RZZ(0.5*pi) q[28],q[27];
RZZ(0.5*pi) q[31],q[33];
RZZ(0.5*pi) q[39],q[34];
U1q(0.131537781476057*pi,0.87447080188528*pi) q[0];
U1q(0.72390371260358*pi,0.6858568798639451*pi) q[1];
U1q(0.693369005904559*pi,1.7042329741276703*pi) q[2];
U1q(0.571660328148682*pi,1.5109199106299984*pi) q[3];
U1q(0.60487522346296*pi,0.04023647196508007*pi) q[4];
U1q(0.162948311142794*pi,1.9438683125835396*pi) q[5];
U1q(0.362756683606393*pi,1.7265509056335842*pi) q[6];
U1q(0.763304041801786*pi,1.7132321613514598*pi) q[7];
U1q(0.305495817692578*pi,1.9349075358854897*pi) q[8];
U1q(0.161594776327082*pi,0.28683556473662986*pi) q[9];
U1q(0.767789454078722*pi,0.09321598059068004*pi) q[10];
U1q(0.209985489700578*pi,1.3954269233322796*pi) q[11];
U1q(0.178133020897086*pi,1.0534621009346*pi) q[12];
U1q(0.0397332919486469*pi,1.6317761073980304*pi) q[13];
U1q(0.42750693841577*pi,0.2765846275798003*pi) q[14];
U1q(0.581747371905651*pi,0.7657390038067717*pi) q[15];
U1q(0.674938656459035*pi,1.4615948538655603*pi) q[16];
U1q(0.709438723621661*pi,1.5251129721972*pi) q[17];
U1q(0.770312811637747*pi,0.3342631761986201*pi) q[18];
U1q(0.287728070570041*pi,0.03896900899261868*pi) q[19];
U1q(0.64353022733798*pi,0.9044343732000901*pi) q[20];
U1q(0.386836058189208*pi,0.33139329102077*pi) q[21];
U1q(0.314721777704478*pi,0.19818430807171006*pi) q[22];
U1q(0.59029448831936*pi,1.1547116115049496*pi) q[23];
U1q(0.664199736826847*pi,0.023504154358794938*pi) q[24];
U1q(0.674353039723298*pi,0.47945095938368*pi) q[25];
U1q(0.802974183533748*pi,0.24877167665254518*pi) q[26];
U1q(0.642958875783546*pi,0.6933107135472201*pi) q[27];
U1q(0.580339592477431*pi,1.3192414477780403*pi) q[28];
U1q(0.305516590816508*pi,1.8946865187205666*pi) q[29];
U1q(0.442895926226715*pi,0.12540113629306138*pi) q[30];
U1q(0.613144851945657*pi,1.9729005763661895*pi) q[31];
U1q(0.278960429695491*pi,1.3999577756663033*pi) q[32];
U1q(0.426359535513994*pi,0.7283295109205481*pi) q[33];
U1q(0.794521983281339*pi,1.72063298431316*pi) q[34];
U1q(0.621391965229616*pi,1.47162042434578*pi) q[35];
U1q(0.362833402836091*pi,1.7322514076510904*pi) q[36];
U1q(0.245063408671306*pi,0.3574011007582145*pi) q[37];
U1q(0.437159843613728*pi,0.3902291264284159*pi) q[38];
U1q(0.0708529102456924*pi,1.5015500099765102*pi) q[39];
RZZ(0.5*pi) q[31],q[0];
RZZ(0.5*pi) q[37],q[1];
RZZ(0.5*pi) q[14],q[2];
RZZ(0.5*pi) q[17],q[3];
RZZ(0.5*pi) q[8],q[4];
RZZ(0.5*pi) q[5],q[18];
RZZ(0.5*pi) q[6],q[24];
RZZ(0.5*pi) q[7],q[28];
RZZ(0.5*pi) q[39],q[9];
RZZ(0.5*pi) q[25],q[10];
RZZ(0.5*pi) q[11],q[23];
RZZ(0.5*pi) q[22],q[12];
RZZ(0.5*pi) q[13],q[36];
RZZ(0.5*pi) q[15],q[19];
RZZ(0.5*pi) q[16],q[38];
RZZ(0.5*pi) q[26],q[20];
RZZ(0.5*pi) q[21],q[29];
RZZ(0.5*pi) q[30],q[27];
RZZ(0.5*pi) q[34],q[32];
RZZ(0.5*pi) q[33],q[35];
U1q(0.240061090490183*pi,0.42983867223478*pi) q[0];
U1q(0.714725052060166*pi,0.7787965570636652*pi) q[1];
U1q(0.275531364038546*pi,0.5447293432315501*pi) q[2];
U1q(0.481158236860641*pi,1.7342615815124685*pi) q[3];
U1q(0.989503931580017*pi,1.03617162781475*pi) q[4];
U1q(0.418496773937023*pi,1.4545697134517201*pi) q[5];
U1q(0.883707201707817*pi,0.8502231389758741*pi) q[6];
U1q(0.563863389313891*pi,1.30502637862992*pi) q[7];
U1q(0.3724180745757*pi,1.1330165581399*pi) q[8];
U1q(0.157871292350909*pi,0.5378627800320599*pi) q[9];
U1q(0.853639246657018*pi,1.6169418720534399*pi) q[10];
U1q(0.202988912683318*pi,1.8515109413789794*pi) q[11];
U1q(0.0699151124486747*pi,1.1692777848612996*pi) q[12];
U1q(0.658613843253774*pi,0.04424718767150004*pi) q[13];
U1q(0.576406914466596*pi,0.37977691413708037*pi) q[14];
U1q(0.0862169995796112*pi,0.5953535130256613*pi) q[15];
U1q(0.596672335819985*pi,1.5805592784890807*pi) q[16];
U1q(0.794873574714168*pi,0.9469008547198898*pi) q[17];
U1q(0.656217628887803*pi,1.291584683681461*pi) q[18];
U1q(0.869741076665462*pi,1.0549536643866988*pi) q[19];
U1q(0.704470304930061*pi,1.8785851122672002*pi) q[20];
U1q(0.448734954028096*pi,0.5978091190750696*pi) q[21];
U1q(0.23701485022509*pi,1.9206020476212498*pi) q[22];
U1q(0.392437462218684*pi,1.47779330265198*pi) q[23];
U1q(0.909936633895047*pi,1.8699915026022147*pi) q[24];
U1q(0.516409816601009*pi,1.7067955861924897*pi) q[25];
U1q(0.48654901792751*pi,0.7061017036805053*pi) q[26];
U1q(0.672079817647325*pi,0.7783503937409901*pi) q[27];
U1q(0.46587902459322*pi,1.1729764861897598*pi) q[28];
U1q(0.409789164390065*pi,0.09282673611442593*pi) q[29];
U1q(0.869589147358018*pi,0.9661935061777909*pi) q[30];
U1q(0.410006533417807*pi,1.0149425633003002*pi) q[31];
U1q(0.534049501586157*pi,1.3397911039534236*pi) q[32];
U1q(0.280699760704203*pi,0.1290813786831384*pi) q[33];
U1q(0.539101125685379*pi,0.7717626145303296*pi) q[34];
U1q(0.793837503876277*pi,0.8960792130994002*pi) q[35];
U1q(0.239563073920944*pi,1.4019594151576094*pi) q[36];
U1q(0.477774719994854*pi,0.5613972766646649*pi) q[37];
U1q(0.447982110661216*pi,0.4186058519601561*pi) q[38];
U1q(0.648993110590599*pi,0.5377681296552899*pi) q[39];
RZZ(0.5*pi) q[12],q[0];
RZZ(0.5*pi) q[1],q[4];
RZZ(0.5*pi) q[28],q[2];
RZZ(0.5*pi) q[3],q[13];
RZZ(0.5*pi) q[5],q[37];
RZZ(0.5*pi) q[6],q[9];
RZZ(0.5*pi) q[7],q[22];
RZZ(0.5*pi) q[38],q[8];
RZZ(0.5*pi) q[10],q[24];
RZZ(0.5*pi) q[11],q[29];
RZZ(0.5*pi) q[26],q[14];
RZZ(0.5*pi) q[15],q[39];
RZZ(0.5*pi) q[16],q[18];
RZZ(0.5*pi) q[17],q[33];
RZZ(0.5*pi) q[19],q[20];
RZZ(0.5*pi) q[21],q[32];
RZZ(0.5*pi) q[30],q[23];
RZZ(0.5*pi) q[25],q[31];
RZZ(0.5*pi) q[27],q[36];
RZZ(0.5*pi) q[34],q[35];
U1q(0.680614473438532*pi,1.9346495055158996*pi) q[0];
U1q(0.614349182933087*pi,0.615706665580074*pi) q[1];
U1q(0.493596683851332*pi,1.2096062208099703*pi) q[2];
U1q(0.376367435117417*pi,1.1680959683537377*pi) q[3];
U1q(0.367393141324943*pi,0.9638040199664104*pi) q[4];
U1q(0.03511539637736*pi,1.3614574668496005*pi) q[5];
U1q(0.357782076035792*pi,0.625115340673644*pi) q[6];
U1q(0.352421301181708*pi,1.0804510335592994*pi) q[7];
U1q(0.248069844067936*pi,0.34189850249179976*pi) q[8];
U1q(0.481051832163197*pi,1.2097990064645003*pi) q[9];
U1q(0.90001355078733*pi,1.4670343543957802*pi) q[10];
U1q(0.595094644113786*pi,0.2616071573976999*pi) q[11];
U1q(0.301948172246042*pi,0.4104687746974598*pi) q[12];
U1q(0.326477382498781*pi,0.33451599921274955*pi) q[13];
U1q(0.348786993613541*pi,1.0518257618935998*pi) q[14];
U1q(0.362897624938399*pi,0.1307548434141701*pi) q[15];
U1q(0.810632074195233*pi,1.1804283615631501*pi) q[16];
U1q(0.358741197148189*pi,0.8970137158799503*pi) q[17];
U1q(0.95091363496088*pi,1.3884498182119493*pi) q[18];
U1q(0.645109860111134*pi,1.278632148676289*pi) q[19];
U1q(0.52775879379014*pi,1.98250788482303*pi) q[20];
U1q(0.385129417920951*pi,1.5930638928382104*pi) q[21];
U1q(0.222296746039325*pi,1.8716670525109809*pi) q[22];
U1q(0.556571432009473*pi,0.31086865322516033*pi) q[23];
U1q(0.17080852979392*pi,0.8815238017539748*pi) q[24];
U1q(0.836160135320754*pi,1.5205829745514006*pi) q[25];
U1q(0.582244038928473*pi,0.6190236747417455*pi) q[26];
U1q(0.218205924523009*pi,1.3888830156551997*pi) q[27];
U1q(0.222797964930288*pi,0.4609805293553908*pi) q[28];
U1q(0.711128691463542*pi,1.675204795293296*pi) q[29];
U1q(0.449533587148861*pi,1.4494025899854206*pi) q[30];
U1q(0.720884371614449*pi,1.3660774006587992*pi) q[31];
U1q(0.377422286890285*pi,1.7347611523259232*pi) q[32];
U1q(0.335114974321357*pi,1.0934827354072691*pi) q[33];
U1q(0.167413944911944*pi,1.3455291325408396*pi) q[34];
U1q(0.491239284331363*pi,1.83439341119107*pi) q[35];
U1q(0.288498170835382*pi,1.4282285099278003*pi) q[36];
U1q(0.350507753963818*pi,0.9513300384831549*pi) q[37];
U1q(0.351610882348318*pi,1.2374895480313377*pi) q[38];
U1q(0.666596259104118*pi,0.9622209162069097*pi) q[39];
rz(2.8819016738908*pi) q[0];
rz(3.2968945621040344*pi) q[1];
rz(1.1706715395739202*pi) q[2];
rz(2.994661628361861*pi) q[3];
rz(1.31665594536899*pi) q[4];
rz(3.3653023345435003*pi) q[5];
rz(1.3148437838139966*pi) q[6];
rz(2.8769573608577*pi) q[7];
rz(3.370265771667899*pi) q[8];
rz(2.4853124716762496*pi) q[9];
rz(2.7775994444324796*pi) q[10];
rz(1.1959259211902005*pi) q[11];
rz(2.59747646622764*pi) q[12];
rz(1.1671141447052396*pi) q[13];
rz(2.0491428842006005*pi) q[14];
rz(3.840648767514029*pi) q[15];
rz(0.3806661356708796*pi) q[16];
rz(2.4013884277574196*pi) q[17];
rz(0.19772601014624946*pi) q[18];
rz(2.9103341888953507*pi) q[19];
rz(2.35335585191714*pi) q[20];
rz(2.69332430809174*pi) q[21];
rz(3.46280300856116*pi) q[22];
rz(2.0565834307651*pi) q[23];
rz(1.1077363227714647*pi) q[24];
rz(2.7768316248142*pi) q[25];
rz(1.6207219112997535*pi) q[26];
rz(2.6114817409695803*pi) q[27];
rz(0.8793652479369101*pi) q[28];
rz(2.777213365287774*pi) q[29];
rz(2.7218143304009086*pi) q[30];
rz(3.536998841007099*pi) q[31];
rz(3.1539866723885464*pi) q[32];
rz(2.9948074919893326*pi) q[33];
rz(3.8280974391860996*pi) q[34];
rz(1.89434920907075*pi) q[35];
rz(0.7262488186681999*pi) q[36];
rz(1.1031194935140043*pi) q[37];
rz(3.7766133343695625*pi) q[38];
rz(1.39991085958847*pi) q[39];
U1q(0.680614473438532*pi,1.816551179406694*pi) q[0];
U1q(1.61434918293309*pi,0.912601227684108*pi) q[1];
U1q(0.493596683851332*pi,1.380277760383886*pi) q[2];
U1q(0.376367435117417*pi,1.162757596715559*pi) q[3];
U1q(0.367393141324943*pi,1.2804599653353979*pi) q[4];
U1q(0.03511539637736*pi,1.726759801393096*pi) q[5];
U1q(1.35778207603579*pi,0.939959124487638*pi) q[6];
U1q(1.35242130118171*pi,0.957408394416985*pi) q[7];
U1q(1.24806984406794*pi,0.71216427415973*pi) q[8];
U1q(1.4810518321632*pi,0.695111478140742*pi) q[9];
U1q(1.90001355078733*pi,1.244633798828258*pi) q[10];
U1q(0.595094644113786*pi,0.457533078587894*pi) q[11];
U1q(0.301948172246042*pi,0.00794524092507726*pi) q[12];
U1q(1.32647738249878*pi,0.501630143917991*pi) q[13];
U1q(1.34878699361354*pi,0.100968646094258*pi) q[14];
U1q(1.3628976249384*pi,0.97140361092822*pi) q[15];
U1q(1.81063207419523*pi,0.561094497234034*pi) q[16];
U1q(0.358741197148189*pi,0.298402143637371*pi) q[17];
U1q(0.95091363496088*pi,0.586175828358197*pi) q[18];
U1q(1.64510986011113*pi,1.18896633757164*pi) q[19];
U1q(0.52775879379014*pi,1.335863736740166*pi) q[20];
U1q(3.3851294179209512*pi,1.286388200929947*pi) q[21];
U1q(0.222296746039325*pi,0.334470061072138*pi) q[22];
U1q(0.556571432009473*pi,1.36745208399027*pi) q[23];
U1q(0.17080852979392*pi,0.989260124525436*pi) q[24];
U1q(1.83616013532075*pi,1.297414599365603*pi) q[25];
U1q(1.58224403892847*pi,1.23974558604155*pi) q[26];
U1q(1.21820592452301*pi,1.0003647566247809*pi) q[27];
U1q(0.222797964930288*pi,0.34034577729229*pi) q[28];
U1q(1.71112869146354*pi,1.45241816058108*pi) q[29];
U1q(1.44953358714886*pi,1.171216920386323*pi) q[30];
U1q(0.720884371614449*pi,1.903076241665961*pi) q[31];
U1q(1.37742228689028*pi,1.888747824714429*pi) q[32];
U1q(1.33511497432136*pi,1.088290227396574*pi) q[33];
U1q(0.167413944911944*pi,0.17362657172694*pi) q[34];
U1q(0.491239284331363*pi,0.728742620261825*pi) q[35];
U1q(1.28849817083538*pi,1.154477328596002*pi) q[36];
U1q(0.350507753963818*pi,1.054449531997159*pi) q[37];
U1q(1.35161088234832*pi,0.0141028824008088*pi) q[38];
U1q(0.666596259104118*pi,1.362131775795381*pi) q[39];
RZZ(0.5*pi) q[12],q[0];
RZZ(0.5*pi) q[1],q[4];
RZZ(0.5*pi) q[28],q[2];
RZZ(0.5*pi) q[3],q[13];
RZZ(0.5*pi) q[5],q[37];
RZZ(0.5*pi) q[6],q[9];
RZZ(0.5*pi) q[7],q[22];
RZZ(0.5*pi) q[38],q[8];
RZZ(0.5*pi) q[10],q[24];
RZZ(0.5*pi) q[11],q[29];
RZZ(0.5*pi) q[26],q[14];
RZZ(0.5*pi) q[15],q[39];
RZZ(0.5*pi) q[16],q[18];
RZZ(0.5*pi) q[17],q[33];
RZZ(0.5*pi) q[19],q[20];
RZZ(0.5*pi) q[21],q[32];
RZZ(0.5*pi) q[30],q[23];
RZZ(0.5*pi) q[25],q[31];
RZZ(0.5*pi) q[27],q[36];
RZZ(0.5*pi) q[34],q[35];
U1q(3.240061090490183*pi,1.31174034612562*pi) q[0];
U1q(1.71472505206017*pi,0.7495113362005164*pi) q[1];
U1q(1.27553136403855*pi,1.7154008828054699*pi) q[2];
U1q(0.481158236860641*pi,1.7289232098743201*pi) q[3];
U1q(0.989503931580017*pi,1.3528275731837498*pi) q[4];
U1q(0.418496773937023*pi,0.81987204799517*pi) q[5];
U1q(3.883707201707817*pi,1.7148513261854061*pi) q[6];
U1q(1.56386338931389*pi,0.7328330493463681*pi) q[7];
U1q(1.3724180745757*pi,0.921046218511625*pi) q[8];
U1q(1.15787129235091*pi,0.3670477045731784*pi) q[9];
U1q(1.85363924665702*pi,1.0947262811706024*pi) q[10];
U1q(0.202988912683318*pi,0.047436862569179894*pi) q[11];
U1q(1.06991511244867*pi,1.766754251088918*pi) q[12];
U1q(3.3413861567462257*pi,1.791898955459243*pi) q[13];
U1q(1.5764069144666*pi,1.7730174938508263*pi) q[14];
U1q(3.9137830004203873*pi,1.5068049413167737*pi) q[15];
U1q(3.403327664180015*pi,1.1609635803081022*pi) q[16];
U1q(0.794873574714168*pi,0.34828928247731006*pi) q[17];
U1q(0.656217628887803*pi,0.4893106938277101*pi) q[18];
U1q(1.86974107666546*pi,1.4126448218612309*pi) q[19];
U1q(1.70447030493006*pi,1.23194096418434*pi) q[20];
U1q(3.551265045971903*pi,0.2816429746930853*pi) q[21];
U1q(1.23701485022509*pi,0.38340505618242005*pi) q[22];
U1q(1.39243746221868*pi,0.534376733417099*pi) q[23];
U1q(1.90993663389505*pi,1.9777278253736847*pi) q[24];
U1q(3.483590183398991*pi,0.11120198772450063*pi) q[25];
U1q(3.51345098207249*pi,1.1526675571028144*pi) q[26];
U1q(3.327920182352675*pi,1.6108973785389946*pi) q[27];
U1q(1.46587902459322*pi,0.05234173412667009*pi) q[28];
U1q(1.40978916439006*pi,1.0347962197599574*pi) q[29];
U1q(1.86958914735802*pi,1.654426004193947*pi) q[30];
U1q(1.41000653341781*pi,1.55194140430745*pi) q[31];
U1q(3.465950498413843*pi,0.2837178730869214*pi) q[32];
U1q(3.719300239295797*pi,1.0526915841206916*pi) q[33];
U1q(0.539101125685379*pi,0.59986005371643*pi) q[34];
U1q(0.793837503876277*pi,0.7904284221701601*pi) q[35];
U1q(3.760436926079056*pi,1.1807464233661495*pi) q[36];
U1q(1.47777471999485*pi,1.66451677017867*pi) q[37];
U1q(1.44798211066122*pi,1.8329865784719457*pi) q[38];
U1q(1.6489931105906*pi,0.9376789892437598*pi) q[39];
RZZ(0.5*pi) q[31],q[0];
RZZ(0.5*pi) q[37],q[1];
RZZ(0.5*pi) q[14],q[2];
RZZ(0.5*pi) q[17],q[3];
RZZ(0.5*pi) q[8],q[4];
RZZ(0.5*pi) q[5],q[18];
RZZ(0.5*pi) q[6],q[24];
RZZ(0.5*pi) q[7],q[28];
RZZ(0.5*pi) q[39],q[9];
RZZ(0.5*pi) q[25],q[10];
RZZ(0.5*pi) q[11],q[23];
RZZ(0.5*pi) q[22],q[12];
RZZ(0.5*pi) q[13],q[36];
RZZ(0.5*pi) q[15],q[19];
RZZ(0.5*pi) q[16],q[38];
RZZ(0.5*pi) q[26],q[20];
RZZ(0.5*pi) q[21],q[29];
RZZ(0.5*pi) q[30],q[27];
RZZ(0.5*pi) q[34],q[32];
RZZ(0.5*pi) q[33],q[35];
U1q(1.13153778147606*pi,1.8671082164751196*pi) q[0];
U1q(1.72390371260358*pi,0.6565716590007994*pi) q[1];
U1q(3.306630994095441*pi,1.5558972519093572*pi) q[2];
U1q(0.571660328148682*pi,0.5055815389918399*pi) q[3];
U1q(0.60487522346296*pi,1.3568924173340804*pi) q[4];
U1q(1.16294831114279*pi,0.30917064712699993*pi) q[5];
U1q(1.36275668360639*pi,0.5911790928431078*pi) q[6];
U1q(0.763304041801786*pi,1.1410388320679061*pi) q[7];
U1q(0.305495817692578*pi,0.7229371962572113*pi) q[8];
U1q(0.161594776327082*pi,0.11602048927774655*pi) q[9];
U1q(0.767789454078722*pi,1.5710003897078444*pi) q[10];
U1q(0.209985489700578*pi,1.59135284452248*pi) q[11];
U1q(3.821866979102914*pi,1.8825699350156224*pi) q[12];
U1q(3.9602667080513565*pi,1.204370035732712*pi) q[13];
U1q(0.42750693841577*pi,1.6698252072935484*pi) q[14];
U1q(3.418252628094349*pi,1.3364194505356641*pi) q[15];
U1q(3.3250613435409653*pi,1.2799280049316262*pi) q[16];
U1q(0.709438723621661*pi,1.92650139995462*pi) q[17];
U1q(0.770312811637747*pi,0.5319891863448598*pi) q[18];
U1q(3.287728070570041*pi,0.39666016646715696*pi) q[19];
U1q(3.356469772662019*pi,0.2060917032514425*pi) q[20];
U1q(3.613163941810792*pi,0.5480588027473854*pi) q[21];
U1q(1.31472177770448*pi,1.1058227957319606*pi) q[22];
U1q(3.40970551168064*pi,1.8574584245641321*pi) q[23];
U1q(3.335800263173152*pi,0.8242151736171113*pi) q[24];
U1q(3.674353039723298*pi,0.33854661453330825*pi) q[25];
U1q(3.197025816466253*pi,0.6099975841307705*pi) q[26];
U1q(3.642958875783546*pi,0.6959370587327673*pi) q[27];
U1q(1.58033959247743*pi,1.9060767725383916*pi) q[28];
U1q(0.305516590816508*pi,0.8366560023660974*pi) q[29];
U1q(1.44289592622671*pi,0.813633634309211*pi) q[30];
U1q(3.3868551480543427*pi,0.5939833912416046*pi) q[31];
U1q(3.721039570304509*pi,0.2235512013740446*pi) q[32];
U1q(3.573640464486006*pi,0.4534434518832815*pi) q[33];
U1q(1.79452198328134*pi,1.54873042349926*pi) q[34];
U1q(0.621391965229616*pi,1.3659696334165399*pi) q[35];
U1q(3.637166597163909*pi,1.8504544308726643*pi) q[36];
U1q(3.754936591328694*pi,0.8685129460851302*pi) q[37];
U1q(0.437159843613728*pi,1.8046098529402066*pi) q[38];
U1q(3.9291470897543075*pi,0.9738971089225417*pi) q[39];
RZZ(0.5*pi) q[26],q[0];
RZZ(0.5*pi) q[25],q[1];
RZZ(0.5*pi) q[18],q[2];
RZZ(0.5*pi) q[3],q[29];
RZZ(0.5*pi) q[7],q[4];
RZZ(0.5*pi) q[38],q[5];
RZZ(0.5*pi) q[14],q[6];
RZZ(0.5*pi) q[30],q[8];
RZZ(0.5*pi) q[13],q[9];
RZZ(0.5*pi) q[15],q[10];
RZZ(0.5*pi) q[17],q[11];
RZZ(0.5*pi) q[21],q[12];
RZZ(0.5*pi) q[16],q[22];
RZZ(0.5*pi) q[19],q[35];
RZZ(0.5*pi) q[37],q[20];
RZZ(0.5*pi) q[23],q[32];
RZZ(0.5*pi) q[36],q[24];
RZZ(0.5*pi) q[28],q[27];
RZZ(0.5*pi) q[31],q[33];
RZZ(0.5*pi) q[39],q[34];
U1q(3.047805427790109*pi,0.6044738621042294*pi) q[0];
U1q(3.375257464422551*pi,0.5355057777879679*pi) q[1];
U1q(1.4631342515629*pi,0.973084477184563*pi) q[2];
U1q(0.407552326551053*pi,0.6418680976862299*pi) q[3];
U1q(1.55768509115386*pi,0.37141716390927026*pi) q[4];
U1q(3.226912161550472*pi,1.8367837790691652*pi) q[5];
U1q(3.538979248791002*pi,1.5091835083040972*pi) q[6];
U1q(1.63267119061332*pi,1.803676176914463*pi) q[7];
U1q(1.3688605387555*pi,1.446819211961941*pi) q[8];
U1q(1.58924232251403*pi,1.2212694314481765*pi) q[9];
U1q(3.201441349352272*pi,1.8389543626900045*pi) q[10];
U1q(0.213991364911671*pi,1.2339040126285798*pi) q[11];
U1q(1.39777210912077*pi,1.373694449305002*pi) q[12];
U1q(3.204315950755605*pi,0.24923490211385313*pi) q[13];
U1q(1.20081692535361*pi,1.1117478674353882*pi) q[14];
U1q(3.299490389701247*pi,0.7686569477010394*pi) q[15];
U1q(3.426615738413052*pi,0.5728526837502561*pi) q[16];
U1q(0.40798292914069*pi,0.27271237616163013*pi) q[17];
U1q(1.57862210362788*pi,0.8212632751573001*pi) q[18];
U1q(3.198496705590518*pi,0.08081521683992998*pi) q[19];
U1q(3.26830239837055*pi,1.4586381064413154*pi) q[20];
U1q(1.20453940153913*pi,0.7800282635695366*pi) q[21];
U1q(0.276278931207435*pi,0.2569693695090347*pi) q[22];
U1q(3.639638613449092*pi,1.1464972744892126*pi) q[23];
U1q(3.411788395048931*pi,0.8521894332226092*pi) q[24];
U1q(3.751832998165425*pi,0.03462238215921776*pi) q[25];
U1q(1.79968771331952*pi,0.6730134309146165*pi) q[26];
U1q(0.489191018509354*pi,1.6553597684775763*pi) q[27];
U1q(1.69972812833699*pi,1.932637257953422*pi) q[28];
U1q(1.68147494864982*pi,0.7883171929650374*pi) q[29];
U1q(1.20630984294307*pi,0.2742142809065108*pi) q[30];
U1q(3.130179405232124*pi,1.9575506022380944*pi) q[31];
U1q(3.176769775596458*pi,1.1440472041726744*pi) q[32];
U1q(3.6232987870380198*pi,1.1526661370971611*pi) q[33];
U1q(3.893408079011916*pi,1.9485955987302033*pi) q[34];
U1q(3.484629817125044*pi,0.54197356804299*pi) q[35];
U1q(1.83523360751031*pi,1.583192694328471*pi) q[36];
U1q(1.32219005315446*pi,0.4997221633436939*pi) q[37];
U1q(0.291657888597398*pi,0.5108584995882266*pi) q[38];
U1q(3.312161268303703*pi,1.5646048349915516*pi) q[39];
RZZ(0.5*pi) q[11],q[0];
RZZ(0.5*pi) q[3],q[1];
RZZ(0.5*pi) q[31],q[2];
RZZ(0.5*pi) q[24],q[4];
RZZ(0.5*pi) q[5],q[29];
RZZ(0.5*pi) q[36],q[6];
RZZ(0.5*pi) q[7],q[17];
RZZ(0.5*pi) q[27],q[8];
RZZ(0.5*pi) q[15],q[9];
RZZ(0.5*pi) q[13],q[10];
RZZ(0.5*pi) q[12],q[35];
RZZ(0.5*pi) q[14],q[20];
RZZ(0.5*pi) q[16],q[21];
RZZ(0.5*pi) q[23],q[18];
RZZ(0.5*pi) q[39],q[19];
RZZ(0.5*pi) q[22],q[33];
RZZ(0.5*pi) q[25],q[34];
RZZ(0.5*pi) q[26],q[30];
RZZ(0.5*pi) q[37],q[28];
RZZ(0.5*pi) q[38],q[32];
U1q(1.71801087598312*pi,0.7852416068350143*pi) q[0];
U1q(3.247902892341607*pi,0.40814940943757483*pi) q[1];
U1q(0.0838540770340036*pi,1.356046887839093*pi) q[2];
U1q(0.700425895737418*pi,1.2685742787792798*pi) q[3];
U1q(1.26270066992672*pi,1.10744759131551*pi) q[4];
U1q(1.53141685100264*pi,1.4155368851704022*pi) q[5];
U1q(1.33331081149553*pi,1.5722877957988413*pi) q[6];
U1q(1.72777515337944*pi,1.1097443839141352*pi) q[7];
U1q(1.01715157460482*pi,0.5427413732630324*pi) q[8];
U1q(1.54597455833632*pi,1.5533134257895023*pi) q[9];
U1q(1.30686526343779*pi,0.37598398219159357*pi) q[10];
U1q(0.65660443924699*pi,0.9788610544452503*pi) q[11];
U1q(0.402533352773335*pi,0.39540936698413187*pi) q[12];
U1q(1.54292292379631*pi,0.5709278683845587*pi) q[13];
U1q(1.37120767866703*pi,0.30169080101485335*pi) q[14];
U1q(0.273398189750861*pi,0.1242296314607092*pi) q[15];
U1q(1.49802151248513*pi,0.6708004786464206*pi) q[16];
U1q(0.825421834392657*pi,1.4443436469411104*pi) q[17];
U1q(1.32770096947093*pi,1.8858907229566473*pi) q[18];
U1q(3.761899077608459*pi,1.1667650097457063*pi) q[19];
U1q(3.450546942648372*pi,1.5601041676626926*pi) q[20];
U1q(0.274176072840521*pi,0.6978760731287563*pi) q[21];
U1q(0.895249285060496*pi,1.8086439507202345*pi) q[22];
U1q(0.241445008826735*pi,1.2983260954672624*pi) q[23];
U1q(1.18030295417209*pi,1.007128052216491*pi) q[24];
U1q(1.24426390671029*pi,1.9942574453664612*pi) q[25];
U1q(0.376714627867173*pi,1.4019719145054665*pi) q[26];
U1q(0.532491925067819*pi,0.0479723902072299*pi) q[27];
U1q(1.61214625516883*pi,1.9600702664990575*pi) q[28];
U1q(1.5852480211268*pi,0.31171415063291863*pi) q[29];
U1q(0.381881296612801*pi,1.4383460719696037*pi) q[30];
U1q(1.66317332742233*pi,1.393476567162005*pi) q[31];
U1q(1.75674276817185*pi,0.5611282792402958*pi) q[32];
U1q(1.30756349548356*pi,1.1255972244848644*pi) q[33];
U1q(0.685512688850475*pi,1.5081514182095832*pi) q[34];
U1q(1.88105505538351*pi,0.4239654867346552*pi) q[35];
U1q(0.330227396893231*pi,0.288051117330991*pi) q[36];
U1q(0.411610081894561*pi,0.470226060673284*pi) q[37];
U1q(0.10568776609996*pi,1.3212015588809267*pi) q[38];
U1q(3.466077170324656*pi,0.9123555137748403*pi) q[39];
rz(3.2147583931649857*pi) q[0];
rz(1.5918505905624252*pi) q[1];
rz(2.643953112160907*pi) q[2];
rz(0.7314257212207202*pi) q[3];
rz(0.89255240868449*pi) q[4];
rz(0.5844631148295978*pi) q[5];
rz(2.4277122042011587*pi) q[6];
rz(0.8902556160858648*pi) q[7];
rz(1.4572586267369676*pi) q[8];
rz(0.44668657421049773*pi) q[9];
rz(1.6240160178084064*pi) q[10];
rz(3.0211389455547497*pi) q[11];
rz(1.6045906330158681*pi) q[12];
rz(3.4290721316154413*pi) q[13];
rz(3.6983091989851467*pi) q[14];
rz(1.8757703685392908*pi) q[15];
rz(1.3291995213535794*pi) q[16];
rz(0.5556563530588896*pi) q[17];
rz(2.1141092770433527*pi) q[18];
rz(0.8332349902542938*pi) q[19];
rz(2.4398958323373074*pi) q[20];
rz(3.3021239268712437*pi) q[21];
rz(2.1913560492797655*pi) q[22];
rz(2.7016739045327376*pi) q[23];
rz(2.992871947783509*pi) q[24];
rz(0.005742554633538788*pi) q[25];
rz(2.5980280854945335*pi) q[26];
rz(3.9520276097927702*pi) q[27];
rz(0.03992973350094253*pi) q[28];
rz(3.688285849367081*pi) q[29];
rz(2.5616539280303963*pi) q[30];
rz(2.606523432837995*pi) q[31];
rz(3.438871720759704*pi) q[32];
rz(0.8744027755151356*pi) q[33];
rz(2.491848581790417*pi) q[34];
rz(1.5760345132653448*pi) q[35];
rz(3.711948882669009*pi) q[36];
rz(1.529773939326716*pi) q[37];
rz(2.6787984411190733*pi) q[38];
rz(1.0876444862251597*pi) q[39];
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