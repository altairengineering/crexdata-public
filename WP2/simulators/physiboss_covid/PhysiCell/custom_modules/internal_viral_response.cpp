#include "./internal_viral_response.h" 

using namespace PhysiCell; 

std::string internal_virus_response_version = "0.6.0"; 

Submodel_Information internal_virus_response_model_info; 

void internal_virus_response_model_setup( void )
{
	// set up the model 
		// set version info 
	internal_virus_response_model_info.name = "internal viral response"; 
	internal_virus_response_model_info.version = internal_virus_response_version; 
		// set functions 
	internal_virus_response_model_info.main_function = NULL; 
	internal_virus_response_model_info.phenotype_function = internal_virus_response_model; 
	internal_virus_response_model_info.mechanics_function = NULL; 
	
		// what microenvironment variables do you expect? 
		// what custom data do I need? 
	internal_virus_response_model_info.cell_variables.push_back( "max_infected_apoptosis_rate" ); 
	internal_virus_response_model_info.cell_variables.push_back( "max_apoptosis_half_max" ); 
	internal_virus_response_model_info.cell_variables.push_back( "apoptosis_hill_power" ); 	
	
		// register the submodel  
	internal_virus_response_model_info.register_model();	
		// set functions for the corresponding cell definition 
		
//	pCD = find_cell_definition( "lung epithelium" ); 
//	pCD->functions.update_phenotype = epithelium_submodel_info.phenotype_function;
//	pCD->functions.custom_cell_rule = epithelium_submodel_info.mechanics_function;
	
	return; 
}

void internal_virus_response_model( Cell* pCell, Phenotype& phenotype, double dt )
{
	static Cell_Definition* pCD = find_cell_definition( "lung epithelium" ); 
	
	// bookkeeping -- find microenvironment variables we need
	static int chemokine_index = microenvironment.find_density_index( "chemokine" );
	static int nINF1 = microenvironment.find_density_index( "interferon 1" );
	
	static int nA_internal = pCell->custom_data.find_variable_index( "assembled_virion" );
	 
	
	// actual model goes here 

	// boolean-controlled apoptosis
	
	static int apoptosis_model_index = cell_defaults.phenotype.death.find_death_model_index( "apoptosis" );
	pCell->phenotype.intracellular->set_boolean_variable_value(
		"Virus_expression", 
		// fuzzy_heavyside_normal(pCell->custom_data[nA_internal], 50, 10)
		pCell->custom_data[nA_internal] > pCell->custom_data["virus_expression_threshold"]
	);
	
	// base death rate (from cell line)
	// double base_death_rate = 
	// 	pCD->phenotype.death.rates[apoptosis_model_index]; 
	
	// // additional death rate from infectoin  
	// double additional_death_rate = pCell->custom_data["max_infected_apoptosis_rate"] ; 
	
	
	// double v = pCell->custom_data[nA_internal] / 
	// 	pCell->custom_data["max_apoptosis_half_max"] ; 
	// v = pow( v, pCell->custom_data["apoptosis_hill_power"] ); 
	
	// double effect = v / (1.0+v); 
	// additional_death_rate *= effect; 
	// phenotype.death.rates[apoptosis_model_index] = base_death_rate + additional_death_rate; 
	
	pCell->phenotype.intracellular->set_boolean_variable_value(
		"Virus_expression", 
		// fuzzy_heavyside_normal(pCell->custom_data[nA_internal], 50, 10)
		pCell->custom_data[nA_internal] > pCell->custom_data["virus_expression_threshold"]
	);
	
	if ( pCell->phenotype.intracellular->get_boolean_variable_value("Apoptosis_type_II") && !pCell->phenotype.death.dead) {
		// std::cout << "Triggering apoptosis based on boolean model" << std::endl;
		pCell->start_death(apoptosis_model_index);
	}
	
	
	
	// if we're infected, secrete a chemokine for the immune model
	double AV = pCell->custom_data[nA_internal];  
	
	static int proinflammatory_cytokine_index = microenvironment.find_density_index( "pro-inflammatory cytokine");
		
	static int nR = pCell->custom_data.find_variable_index( "viral_RNA");
	double R = pCell->custom_data[nR];
	static int antibody_index = microenvironment.find_density_index( "Ig");
	
	
	// FIONA ADDED FOLLOWING 4 line 28th April
	if( pCell->custom_data["cell_virus_induced_apoptosis_flag"]>=1 )
	{
		pCell->custom_data["cell_apo_time"]=pCell->custom_data["cell_apo_time"]+dt;
	}
	//
	
	if( R >= 1.00 - 1e-16 ) 
	{
		pCell->custom_data["infected_cell_chemokine_secretion_activated"] = 1.0; 
		pCell->phenotype.intracellular->set_boolean_variable_value("Chemokin_secretion", true);
		// (AJ-V5) Antibody binding starts once the cell is infected
		double uptakerate = parameters.doubles("Antibody_binding_rate")*(1000-phenotype.molecular.internalized_total_substrates[antibody_index])/1000;
		if( uptakerate < 0)
		{
			uptakerate=0;
		}
		pCell->phenotype.secretion.uptake_rates[antibody_index]=uptakerate; 
	}

	if( pCell->phenotype.intracellular->get_boolean_variable_value("Chemokin_secretion") && phenotype.death.dead == false )
	{
		double rate = AV; 
		rate /= pCell->custom_data["max_apoptosis_half_max"];
		if( rate > 1.0 )
		{ rate = 1.0; }
		rate *= pCell->custom_data[ "infected_cell_chemokine_secretion_rate" ];

		phenotype.secretion.secretion_rates[chemokine_index] = rate; 
		phenotype.secretion.saturation_densities[chemokine_index] = 1.0;

		// (Adrianne) adding pro-inflammatory cytokine secretion by infected cells
		pCell->phenotype.secretion.secretion_rates[proinflammatory_cytokine_index] = pCell->custom_data["activated_cytokine_secretion_rate"];
	}
	
	// (Adrianne) check whether the cell is undergoing pyroptosis and if so, evaluate the pyropotosis model
	if( pCell->custom_data["cell_pyroptosis_flag"]==0.5 )
	{
		pyroptosis_cascade( pCell, phenotype, dt ); 
		return;
	};

	// Sara&Fiona: Internalise pro-pyroptotic cytokine from the microenvironment
	static int pro_pyroptotic_cytokine = microenvironment.find_density_index("pro-pyroptosis cytokine"); 
	double pyroptotic_cytokine_concentration = phenotype.molecular.internalized_total_substrates[pro_pyroptotic_cytokine]; 
	double RNAlow=parameters.doubles("RNA_lower_bound");

    // (Sara&Fiona) pyroptosis cascade in the cell is initiated if cell's viral_RNA is >1 (i.e. >=3). This is arbitraty to check things work.
	
	// FIONA: NOTE IN OUR OURS IT IS R>2 and I added cell_bystander check
	if( R>=RNAlow && (int)pCell->custom_data["cell_pyroptosis_flag"]==0 && (int)pCell->custom_data["cell_virus_induced_apoptosis_flag"]==0 && (int)pCell->custom_data["cell_bystander_pyroptosis_flag"]==0)
	{
		// set the probability (in 0,1) that a cell with a death-sentence pyroptoses (not apoptoses)
		// FIONA THIS IS DIFFERENT FROM OURS AS WE SET THE PROBABILITY
		double cell_death_pyroptosis_probability = (R-RNAlow)/(1000-RNAlow); 
		if( cell_death_pyroptosis_probability > 1.0 )
		{ cell_death_pyroptosis_probability = 1.0; } 
		// FIONA FOR SOME REASON THEY DIVIDE BY 3
		cell_death_pyroptosis_probability/=3; 
		// randomise a number in 0,1 that determines the cell death mode (pyroptosis or apoptosis)
		if(UniformRandom() < cell_death_pyroptosis_probability) 
		{
			pCell->custom_data["cell_pyroptosis_flag"]=1; //cell pyroptoses
			pCell->custom_data["cell_pyroptosis_time"]=1; // FIONA ADDED PYROPTOSIS TIME INITIATED
		}
		else // FIONA ADDED IN ELSE
		{
			pCell->custom_data["cell_virus_induced_apoptosis_flag"]=1; //cell apoptoses
			phenotype.death.rates[apoptosis_model_index] = 1; 
			
		}
		
		return;
	}
	// (Sara&Fiona)  FIONA added cell_bystander check
    else if(R<RNAlow && pyroptotic_cytokine_concentration>100.0 && (int)pCell->custom_data["cell_pyroptosis_flag"]==0 && (int)pCell->custom_data["cell_virus_induced_apoptosis_flag"]==0 && (int)pCell->custom_data["cell_bystander_pyroptosis_flag"]==0) 
    {
		pCell->custom_data["cell_pyroptosis_flag"]=2; // Pyroptosis bystander cascade is initiated
		pCell->custom_data["cell_bystander_pyroptosis_flag"]=1; // Pyroptosis cascade is initiated
		//printf("Pyro bystander effect!\n");
		pCell->custom_data["cell_pyroptosis_time"]=1; // FIONA TIME INITIATED
		return;
    }
	
	// interferon signaling 

	// // approximate activation by extracellular interferons 
	// // // activation = min( 1 , extracellular_interferon / interferon_max_response_threshold ) 
	pCell->custom_data["interferon_activation"] = pCell->nearest_density_vector()[nINF1] / 
		( pCell->custom_data["interferon_max_response_threshold"] + 1e-32 ); 
	if( pCell->custom_data["interferon_activation"] > 1.0 )
	{ pCell->custom_data["interferon_activation"] = 1.0; } 
		
	// // Type-I interferon secretion 
	// // // secretion_rate = r_viral * Heaviside( RNA - 1 ) + r_paracrine * activation 
	phenotype.secretion.secretion_rates[nINF1] = pCell->custom_data["interferon_activation"]; 
	phenotype.secretion.secretion_rates[nINF1] *= pCell->custom_data["max_interferon_secretion_rate_via_paracrine"]; 
	if( R >= pCell->custom_data["interferon_viral_RNA_detection"] - 1e-16 ) // if there is at least 1 complete set of uncoated viral RNA
	{
		double scaled_RNA = (R-pCell->custom_data["interferon_viral_RNA_detection"]) / ( -pCell->custom_data["interferon_viral_RNA_detection"] + pCell->custom_data["interferon_viral_RNA_threshold"] );
		if( scaled_RNA > 1 )
		{ scaled_RNA = 1.0; }
		phenotype.secretion.secretion_rates[nINF1] += pCell->custom_data["interferon_secretion_rate_via_infection"] * scaled_RNA; 
	} 
	
	// // now the interferon response 
	// // // protein_synthesis_rate = protein_synthesis_rate_0 * ( 1 - interferon_activation * interferon_max_virus_inhibition ) 
	pCell->custom_data["protein_synthesis_rate"] = pCell->custom_data["interferon_max_virus_inhibition"]; // inhibition
	pCell->custom_data["protein_synthesis_rate"] *= pCell->custom_data["interferon_activation"]; // activation*inhibition
	pCell->custom_data["protein_synthesis_rate"] *= -1; // -activation*inhibition 
	pCell->custom_data["protein_synthesis_rate"] += 1.0; // 1 - activation*inhibition 
	pCell->custom_data["protein_synthesis_rate"] *= pCD->custom_data["protein_synthesis_rate"]; 
	// protein_synthesis_rate0 * (1 - activation*inhibition)
	
	return; 	
}
// Sara&Fiona: code for pyroptosis
void pyroptosis_cascade( Cell* pCell, Phenotype& phenotype, double dt )
{
	// Sara&Fiona: Pyroptosis code starts here.
	// FIONA ADDED IN TIME UPDATER - THIS IS WHERE WE UPDATE HOW LONG THE CELL HAS BEEN UNDERGOING PYROPTOSIS
	pCell->custom_data["cell_pyroptosis_time"]=pCell->custom_data["cell_pyroptosis_time"]+dt;
	
	// Intracellular components
	static int nfkb_n = pCell->custom_data.find_variable_index( "nuclear_NFkB" ); 
	static int nlrp3_i = pCell->custom_data.find_variable_index( "inactive_NLRP3" ); 
	static int nlrp3_a = pCell->custom_data.find_variable_index( "active_NLRP3" ); 
	static int nlrp3_b = pCell->custom_data.find_variable_index( "bound_NLRP3" ); 
	static int asc_b = pCell->custom_data.find_variable_index( "bound_ASC" );
	static int caspase1_b = pCell->custom_data.find_variable_index( "bound_caspase1" );
	static int gsdmd_c = pCell->custom_data.find_variable_index( "cleaved_gasderminD" );
	static int il_1b_p = pCell->custom_data.find_variable_index( "pro_IL_1b" );
	static int il_1b_c = pCell->custom_data.find_variable_index( "cytoplasmic_IL_1b" );
	static int il_1b_e = pCell->custom_data.find_variable_index( "external_IL_1b" );
	static int il_18_c = pCell->custom_data.find_variable_index( "cytoplasmic_IL_18" );
	static int il_18_e = pCell->custom_data.find_variable_index( "external_IL_18" );
	static int volume_c = pCell->custom_data.find_variable_index( "cytoplasmic_volume" );


	// System of pyroptosis equations starts here
    //Model constants (definitions could be moved to xml file)
    static double k_nlrp3_ita =  0.7; // FIONA: VALUE UPDATED FROM 0.07
    static double k_nlrp3_atb = 1; // FIONA: VALUE UPDATED FROM 0.07
    static double k_asc_ftb = 0.04; // FIONA: VALUE UPDATED FROM 0.02
    static double k_c1_ftb = 0.03; // FIONA: VALUE UPDATED FROM 0.04
    static double k_il1b_cte = 1; // FIONA: VALUE UPDATED FROM 0.8
    static double k_il18_cte = 1;// FIONA: VALUE UPDATED FROM 0.8
    static double k_vol_c = 0.2;// FIONA: VALUE UPDATED FROM 0.1
    // Decay constants
    static double d_nlrp3 = 0.002;
    static double d_il =0.004;
    // Hill function rates
    static double a_nlrp3 = 0.07; // FIONA: VALUE UPDATED FROM 0.025
    static double a_il1b_p = 0.06; // FIONA: VALUE UPDATED FROM 0.007
    static double a_gsdmd =0.1; // FIONA: VALUE UPDATED FROM 0.08
    static double a_il1b_c = 1;// FIONA: VALUE UPDATED FROM 0.8
    static double a_il18 = 1;// FIONA: VALUE UPDATED FROM 0.8
    static double hm_nfkb = 0.3;
    static double hm_c1 = 0.3;
    static double hex_nfkb = 2.0;
    static double hex_c1 = 2.0 ;
	
	double no_a=-1; // FIONA: ADDED
	double no_b=2;// FIONA: ADDED
	double no_c=1000; // FIONA: ADDED

	double nf_hh=0.55; // FIONA: ADDED
	double nf_orig=0.25; // FIONA: ADDED
	double nf_ss=0.8; // FIONA: ADDED
	double nf_tt=10; // FIONA: ADDED
	double t_i=pCell->custom_data["cell_pyroptosis_time"]-dt;	// FIONA: ADDED - this is time dependence

	//Update nuclear NFkB (updated backward)
	pCell->custom_data[nfkb_n]= (nf_hh)*exp(-(log((t_i)/nf_tt)*log((t_i)/nf_tt))/nf_ss)+nf_orig; // FIONA: NEW

	//Set Hill function 1
	double hill_nfkb = (pow((pCell->custom_data[nfkb_n]-nf_orig),hex_nfkb))/(pow(hm_nfkb,hex_nfkb)+pow((pCell->custom_data[nfkb_n]-nf_orig),hex_nfkb)); // FIONA: NEW
	
	
	//Update NLRP3 (inactive, active and bound) (updated backward)
	
	pCell->custom_data[nlrp3_i] = ((pCell->custom_data[nlrp3_i])+dt*a_nlrp3*hill_nfkb)/(1+dt*k_nlrp3_ita+dt*d_nlrp3); // FIONA: NEW		
	pCell->custom_data[nlrp3_a]=(-(1+dt*d_nlrp3)+sqrt(((1+dt*d_nlrp3)*(1+dt*d_nlrp3))+4*(dt*k_nlrp3_atb)*(pCell->custom_data[nlrp3_a]+dt*k_nlrp3_ita*pCell->custom_data[nlrp3_i])))/(2*dt*k_nlrp3_atb); // FIONA: NEW
	pCell->custom_data[nlrp3_b] = pCell->custom_data[nlrp3_b] + dt * k_nlrp3_atb * pCell->custom_data[nlrp3_a]* pCell->custom_data[nlrp3_a]; // FIONA: NEW
	
	
	// Define F_no FIONA ADDED
	double F_no = 1/(1+pow(((pCell->custom_data[nlrp3_b]-no_a)/no_b),-no_c)); // define new function for ASC binding

	//Update bound ASC (updated backward)
	pCell->custom_data[asc_b]=(pCell->custom_data[asc_b]+dt*k_asc_ftb*F_no*pCell->custom_data[nlrp3_b])/(1+dt*k_asc_ftb*F_no*pCell->custom_data[nlrp3_b]); // FIONA: NEW	
	

	//Update bound caspase1 (updated backward)
	pCell->custom_data[caspase1_b]=(pCell->custom_data[caspase1_b]+dt*k_c1_ftb*pCell->custom_data[asc_b])/(1+dt*k_c1_ftb*pCell->custom_data[asc_b]); // FIONA: NEW
	

	//Set Hill function 2
	double hill_c1 = (pow(pCell->custom_data[caspase1_b],hex_c1))/(pow(hm_c1,hex_c1)+pow(pCell->custom_data[caspase1_b],hex_c1)); // FIONA: NEW
	
	//Update cleaved GSDMD (updated backward)
	pCell->custom_data[gsdmd_c] = (pCell->custom_data[gsdmd_c]+dt*a_gsdmd*hill_c1)/(1+dt*a_gsdmd*hill_c1); // FIONA: NEW


	//Set G function (same now that total GSDMD concentration is 1 au of concentration)
	double g_gsdmd = pCell->custom_data[gsdmd_c]/1;

	//Update IL1b (pro, cytoplasmic, external)	We want to relate this to secreted cytokine IL1b (updated backward)
	
	pCell->custom_data[il_1b_p] = (pCell->custom_data[il_1b_p]+dt*a_il1b_p*hill_nfkb)/(1+dt*a_il1b_c*hill_c1+dt*d_il); //FIONA:NEW
	pCell->custom_data[il_1b_c] = (pCell->custom_data[il_1b_c]+dt*a_il1b_c*hill_c1*(pCell->custom_data[il_1b_p]))/(1+dt*d_il+dt*k_il1b_cte*g_gsdmd); //FIONA:NEW		
	pCell->custom_data[il_1b_e] = pCell->custom_data[il_1b_e] + dt * (k_il1b_cte*g_gsdmd*pCell->custom_data[il_1b_c]); //FIONA:NEW


	//Update IL18 (cytoplasmic, external)(updated backward)
	
	pCell->custom_data[il_18_c] = (pCell->custom_data[il_18_c]+dt*a_il18*hill_c1*(1-pCell->custom_data[il_18_e]))/((1+dt*a_il18*hill_c1)*(1+dt*k_il18_cte*g_gsdmd)); //FIONA:NEW
	pCell->custom_data[il_18_e] = pCell->custom_data[il_18_e] +  dt * k_il18_cte*g_gsdmd*pCell->custom_data[il_18_c]; //FIONA:NEW


	//Update cytoplasmic volume (updated backward) SAME
	pCell->custom_data[volume_c] = pCell->custom_data[volume_c]/(1-dt * k_vol_c * g_gsdmd);
	

	// (Yafei) need to update the real radius SAME
	phenotype.volume.total = pCell->custom_data[volume_c]; 

	//Temporary: "super fast" apoptosis occurs when cell should burst. 
	//To do: We actually want the cell to rupture once a cytoplasmic critical volume is reached (e.g. 1.5 of initial cytoplasmic volume from in vitro data). 
	static int apoptosis_model_index = cell_defaults.phenotype.death.find_death_model_index( "apoptosis" );	
	static double initial_total_volume = 2494;

	// if( pCell->custom_data[volume_c] > 1.2*initial_total_volume )
	// {
	// 	//std::cout<<"Pyroptotic cell burst!"<<std::endl;
	// 	//The cell's 'apoptosis death rate' is set to be "super high" 
	// 	phenotype.death.rates[apoptosis_model_index] = 9e9; 
	// }
	
	
	// Here is how we plug the typeII apoptosis. Maybe we should have a node linked to cell volume ?
	// if ( pCell->phenotype.intracellular->get_boolean_variable_value("Apoptosis_type_II") && !pCell->phenotype.death.dead) {
		// std::cout << "Triggering apoptosis based on boolean model" << std::endl;
		// pCell->start_death(apoptosis_model_index);
	// }
	
	// Here I'm not sure why I was doing this. It should already be set by viral_dynamics
	// pCell->phenotype.intracellular->set_boolean_node_value("Virus_inside", R >= 1.00 - 1e-16);

	// Here I activate (in one go) the chemokin secretion. At that time there was only one cytokin
	if (pCell->phenotype.intracellular->get_boolean_variable_value("Chemokin_secretion") && !phenotype.death.dead)
	// if( pCell->custom_data["infected_cell_chemokine_secretion_activated"] > 0.1 && phenotype.death.dead == false )
	{	
		// (Adrianne) update cell pro-inflammatory secretion rate based on IL18 secretion rate - need to double check unit conversion
		static int proinflammatory_cytokine_index = microenvironment.find_density_index( "pro-inflammatory cytokine");
		pCell->phenotype.secretion.secretion_rates[proinflammatory_cytokine_index] = pCell->custom_data["activated_cytokine_secretion_rate"]+k_il18_cte*g_gsdmd*pCell->custom_data[il_18_c];
		// (Sara and Fiona)
		static int propyroptotic_cytokine_index = microenvironment.find_density_index("pro-pyroptosis cytokine");
		pCell->phenotype.secretion.secretion_rates[propyroptotic_cytokine_index] = k_il1b_cte*g_gsdmd*pCell->custom_data[il_1b_c];
	}
	
	return; 
}
