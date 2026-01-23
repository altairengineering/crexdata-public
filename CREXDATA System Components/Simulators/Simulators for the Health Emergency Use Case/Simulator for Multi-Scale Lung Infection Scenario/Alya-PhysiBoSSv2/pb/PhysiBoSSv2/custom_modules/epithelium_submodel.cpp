#include "./epithelium_submodel.h" 

using namespace PhysiCell; 

std::string epithelium_submodel_version = "0.6.0"; 

Submodel_Information epithelium_submodel_info; 

void create_secreting_agent( Cell_Definition* pCD, double positionpass0, double positionpass1)
{
	std::vector<double> positionpass = {0,0,0}; 
	positionpass[0]=positionpass0;
	positionpass[1]=positionpass1;
	
	Cell* pC = create_cell( *pCD );
	
	pC->assign_position( positionpass );
	pC->is_movable = false; 
	
	return; 
}

void create_secreting_agentcall(double positionpass0, double positionpass1)
{
	static Cell_Definition* pCD = find_cell_definition( "residual" );
	create_secreting_agent( pCD, positionpass0, positionpass1 ); 
	
	return;
}

void epithelium_contact_function( Cell* pC1, Phenotype& p1, Cell* pC2, Phenotype& p2, double dt )
{
	// elastic adhesions 
	standard_elastic_contact_function( pC1,p1, pC2, p2, dt );
	
	return; 
}
void custom_update_cell_and_death_parameters_O2_based( Cell* pCell, Phenotype& phenotype, double dt )
{
	static int	start_phase_index = phenotype.cycle.model().find_phase_index( PhysiCell_constants::G0G1_phase );
	static int	necrosis_index = phenotype.death.find_death_model_index( PhysiCell_constants::necrosis_death_model ); 
	static int	end_phase_index = phenotype.cycle.model().find_phase_index( PhysiCell_constants::S_phase );
	static int oxygen_substrate_index = pCell->get_microenvironment()->find_density_index( "oxygen" ); 
	// sample the microenvironment to get the pO2 value 
	
	double pO2 = (pCell->nearest_density_vector())[oxygen_substrate_index]; // PhysiCell_constants::oxygen_index]
	int n = pCell->phenotype.cycle.current_phase_index(); 
	double my_necrosis_threshold = 5.0;
	double my_o2_necrosis_max = 2.5; 
	// this multiplier is for linear interpolation of the oxygen value 
	double multiplier = 1.0;

	if( phenotype.death.dead == true )	
	{ return; }

	if( pO2 < pCell->parameters.o2_proliferation_saturation )
	{
		multiplier = ( pO2 - pCell->parameters.o2_proliferation_threshold ) 
			/ ( pCell->parameters.o2_proliferation_saturation - pCell->parameters.o2_proliferation_threshold );
	}
	if( pO2 < pCell->parameters.o2_proliferation_threshold )
	{ 
		multiplier = 0.0; 
	}
	
	// now, update the appropriate cycle transition rate 
	
	phenotype.cycle.data.transition_rate(start_phase_index,end_phase_index) = multiplier * 
		pCell->parameters.pReference_live_phenotype->cycle.data.transition_rate(start_phase_index,end_phase_index);
	
	// Update necrosis rate
	
	multiplier = 0.0;
	if( pO2 < my_necrosis_threshold )
	{
		multiplier = ( my_necrosis_threshold - pO2 ) 
			/ ( my_necrosis_threshold - my_o2_necrosis_max );
	}
	if( pO2 < my_o2_necrosis_max )
	{ 
		multiplier = 1.0; 
	}	
	
	// now, update the necrosis rate 
	
	pCell->phenotype.death.rates[necrosis_index] = multiplier * pCell->parameters.max_necrosis_rate; 
	
	// check for deterministic necrosis 
	
	// if( pCell->parameters.necrosis_type == PhysiCell_constants::deterministic_necrosis && multiplier > 1e-16 )
	// {	pCell->phenotype.death.rates[necrosis_index] = 9e99;
	// 	std::cout << "This cell " << pCell->ID << " has cycle code: " <<  pCell->phenotype.cycle.model().code<<"and nearest voxel "<<pO2 << std::endl;
	// } 

// 
	return; 




}
void epithelium_phenotype( Cell* pCell, Phenotype& phenotype, double dt )
{
	custom_update_cell_and_death_parameters_O2_based(pCell,phenotype,dt);
	// int necrosis_model_index = cell_defaults.phenotype.death.find_death_model_index( "Necrosis" );
	// static int apoptosis_model_index = phenotype.death.find_death_model_index( "Apoptosis" );
	// static int oxygen_substrate_index = pCell->get_microenvironment()->find_density_index( "oxygen" ); 

	// double oxygen_internal = pCell->phenotype.molecular.internalized_total_substrates[oxygen_substrate_index];
	// double oxygen_external = pCell->nearest_density_vector()[oxygen_substrate_index];

	// float my_custom_threshold = 5.0;
	
	// std::cout << "External O2: "  << oxygen_external << "; Internal O2: " << oxygen_internal << std::endl;


	// if(oxygen_external < my_custom_threshold)
	// {
	// 	// if(oxygen_internal < my_custom_threshold)
	// 	pCell-> phenotype.death.rates[apoptosis_model_index] == 1e10;
	// 	pCell-> phenotype.death.rates[necrosis_model_index] == 1e-10;
	// 	// pCell->phenotype.cycle.data.transition_rate(0,1) = 1e-10;
	// 	pCell->start_death( necrosis_model_index ); 

	// 	// pCell-> phenotype.death.rates[necrosis_model_index] = 9e99; 
	// 	// pCell-> phenotype.death.dead == true;

	// 	std::cout << "Death code: "  << pCell-> phenotype.death.dead << "; Current cycle code: " <<pCell->phenotype.cycle.model().code<< std::endl;
	// }

	//  BN inputs are set, run maboss:
	if(pCell->phenotype.intracellular->need_update())
	{		
		pCell->phenotype.intracellular->update();
	}
	static int debris_index = microenvironment.find_density_index( "debris");

	// receptor dynamics 
	// requires faster time scale - done in main function 
	
	// viral dynamics model 
	internal_viral_dynamics_info.phenotype_function(pCell,phenotype,dt); 
	// internal_virus_model(pCell,phenotype,dt);
	
	// viral response model 
	internal_virus_response_model_info.phenotype_function(pCell,phenotype,dt); 
	// internal_virus_response_model(pCell,phenotype,dt);	

	// T-cell based death
	TCell_induced_apoptosis(pCell, phenotype, dt ); 
// // 
// 	// (Adrianne V5) ROS induced cell death model
	ROS_induced_apoptosis(pCell, phenotype, dt);
// 	std::cout<<" cycle code "<<pCell->phenotype.cycle.model().code<<std::endl;

	// if I am dead, remove all adhesions 
	// static int apoptosis_index = phenotype.death.find_death_model_index( "apoptosis" ); 
	if( phenotype.death.dead == true )
	{
		// detach all attached cells 
		// remove_all_adhesions( pCell ); 
		
		phenotype.secretion.secretion_rates[debris_index] = pCell->custom_data["debris_secretion_rate"];
	}
	
	// if I am dead, don't bother executing this function again 
	if( phenotype.death.dead == true )
	{
		pCell->functions.update_phenotype = NULL;
	}
	// std::cout << "Leaving phenotype function This cell " << pCell->ID << " has cycle code: " <<  pCell->phenotype.cycle.model().code << std::endl;

	return; 
}

void epithelium_mechanics( Cell* pCell, Phenotype& phenotype, double dt )
{
	static int debris_index = microenvironment.find_density_index( "debris");
	
	pCell->is_movable = false; 
	
	// if I'm dead, don't bother 
	if( phenotype.death.dead == true )
	{
		// the cell death functions don't automatically turn off custom functions, 
		// since those are part of mechanics. 
		// remove_all_adhesions( pCell ); 
		
		// Let's just fully disable now. 
		pCell->functions.custom_cell_rule = NULL; 
		pCell->functions.contact_function = NULL; 

		phenotype.secretion.secretion_rates[debris_index] = pCell->custom_data["debris_secretion_rate"]; 
		return; 
	}	
	
	// this is now part of contact_function 
	/*
	// if I'm adhered to something ... 
	if( pCell->state.neighbors.size() > 0 )
	{
		// add the elastic forces 
		extra_elastic_attachment_mechanics( pCell, phenotype, dt );
	}
	*/
	return; 
}

void epithelium_submodel_setup( void )
{
	Cell_Definition* pCD;
	
	// set up any submodels you need 
	// viral replication 
	
	// receptor trafficking 
	receptor_dynamics_model_setup(); // done 
	// viral replication 
	internal_virus_model_setup();	
	// single-cell response 
	internal_virus_response_model_setup(); 
 	
	// set up epithelial cells
		// set version info 
	epithelium_submodel_info.name = "epithelium model"; 
	epithelium_submodel_info.version = epithelium_submodel_version; 
		// set functions 
	epithelium_submodel_info.main_function = NULL; 
	epithelium_submodel_info.phenotype_function = epithelium_phenotype; 
	epithelium_submodel_info.mechanics_function = epithelium_mechanics; 
	
		// what microenvironment variables do you expect? 
	epithelium_submodel_info.microenvironment_variables.push_back( "virion" ); 
	epithelium_submodel_info.microenvironment_variables.push_back( "interferon 1" ); 
	epithelium_submodel_info.microenvironment_variables.push_back( "pro-inflammatory cytokine" ); 
	epithelium_submodel_info.microenvironment_variables.push_back( "chemokine" ); 
	epithelium_submodel_info.microenvironment_variables.push_back( "anti-inflammatory cytokine" );
	epithelium_submodel_info.microenvironment_variables.push_back( "pro-pyroptosis cytokine" );
		// what custom data do I need? 
	//epithelium_submodel_info.cell_variables.push_back( "something" ); 
		// register the submodel  
	epithelium_submodel_info.register_model();	
		// set functions for the corresponding cell definition 
	pCD = find_cell_definition( "lung epithelium" ); 
	pCD->functions.update_phenotype = epithelium_submodel_info.phenotype_function;
	pCD->functions.custom_cell_rule = epithelium_submodel_info.mechanics_function;
	pCD->functions.contact_function = epithelium_contact_function; 
	
	return;
}

void TCell_induced_apoptosis( Cell* pCell, Phenotype& phenotype, double dt )
{
	static int apoptosis_index = phenotype.death.find_death_model_index( "Apoptosis" ); 
	static int debris_index = microenvironment.find_density_index( "debris" ); 
	static int proinflammatory_cytokine_index = microenvironment.find_density_index("pro-inflammatory cytokine");
	static int antiinflammatory_cytokine_index = microenvironment.find_density_index("anti-inflammatory cytokine");
	
	pCell->phenotype.intracellular->set_boolean_variable_value(
		"TCell_attached", 
		pCell->custom_data["TCell_contact_time"] > pCell->custom_data["TCell_contact_death_threshold"]// || pCell->phenotype.intracellular->get_boolean_variable_value("TCell_attached")
	);
	
	if ( pCell->phenotype.intracellular->get_boolean_variable_value("Apoptosis_type_I") && !pCell->phenotype.death.dead )
	
	// if( pCell->custom_data["TCell_contact_time"] > pCell->custom_data["TCell_contact_death_threshold"] )
	{
		// make sure to get rid of all adhesions! 
		// detach all attached cells 
		// remove_all_adhesions( pCell ); 
		
		//#pragma omp critical
		//{
		//std::cout << "\t\t\t\t" << pCell << " (of type " << pCell->type_name <<  ") died from T cell contact" << std::endl; 
		//}
		
		// induce death 
		pCell->start_death( apoptosis_index ); 
		
		pCell->phenotype.secretion.secretion_rates[proinflammatory_cytokine_index] = 0; 
		pCell->phenotype.secretion.secretion_rates[debris_index] = pCell->custom_data["debris_secretion_rate"]; 
		pCell->functions.update_phenotype = NULL; 
		double positionpass0=pCell->position[0];
		double positionpass1=pCell->position[1];
		create_secreting_agentcall(positionpass0, positionpass1);
		
	}
	
	return; 
}

void ROS_induced_apoptosis( Cell* pCell, Phenotype& phenotype, double dt )
{
	static int apoptosis_index = phenotype.death.find_death_model_index( "Apoptosis" ); 
	static int ROS_index = microenvironment.find_density_index( "ROS" ); 
	double ROS_amount = pCell->nearest_density_vector()[ROS_index];

	static int debris_index = microenvironment.find_density_index( "debris" ); 
	static int proinflammatory_cytokine_index = microenvironment.find_density_index("pro-inflammatory cytokine");
	
	double epsilon_ROS = parameters.doubles("epsilon_ROS");
	// double epsilon_ROS 
	double prob_apoptosis = ROS_amount/(ROS_amount+epsilon_ROS);
	// double prob_apoptosis = ROS_amount;

	double prob_random = UniformRandom();

	// if (prob_apoptosis < 0.0001)
	// {prob_apoptosis=0;}
	// else
	// {prob_apoptosis = prob_apoptosis;}
	if( (prob_random < prob_apoptosis) && !pCell->phenotype.death.dead )
	{

		// std::cout<<"prob_random "<<prob_random<<" ROS amount "<<ROS_amount<<" prob apoptosis "<<prob_apoptosis<<" epsilon "<<epsilon_ROS<<std::endl;
		// make sure to get rid of all adhesions! 
		// detach all attached cells 
		// remove_all_adhesions( pCell ); 
		
		//#pragma omp critical
		//{
		//std::cout << "\t\t\t\t" << pCell << " (of type " << pCell->type_name <<  ") died from ROS" << std::endl; 
		//}
		// std::cout << "This cell " << pCell->ID << " has cycle code: " <<  pCell->phenotype.cycle.model().code<< "but wasnt labeled dead" << std::endl;

		// induce death 
		pCell->start_death( apoptosis_index ); 

		pCell->phenotype.secretion.secretion_rates[proinflammatory_cytokine_index] = 0; 
		pCell->phenotype.secretion.secretion_rates[debris_index] = pCell->custom_data["debris_secretion_rate"]; 
		
		pCell->functions.update_phenotype = NULL; 
	}
	
	return; 
}


double total_epithelial_cell_count()
{
	double out = 0.0;
	
	for( int i=0; i < (*all_cells).size() ; i++ )
	{
		if( (*all_cells)[i]->type == 1 )
		{ out += 1.0; } 
	}
	
	return out; 

}
double total_alive_epithelial_cell_count()
{
	double out = 0.0;
	
	for( int i=0; i < (*all_cells).size() ; i++ )
	{
		if( (*all_cells)[i]->phenotype.cycle.current_phase().code==4 && (*all_cells)[i]->type == 1 )
		{ out += 1.0; } 
	}
	
	return out; 

}
double total_apoptotic_epithelial_cell_count()
{
	double out = 0.0;
	int apoptosis_model_index = PhysiCell_constants::apoptosis_death_model;


	for( int i=0; i < (*all_cells).size() ; i++ )
	{
		if((*all_cells)[i]->phenotype.cycle.current_phase().code == 100  && (*all_cells)[i]->type == 1 )
		{ out += 1.0; } 
	}
	// std::cout<<apoptosis_model_index<<std::endl;
	// out=apoptotic_dead-out;

	return out; 

}
double total_necrotic_epithelial_cell_count()
{
	double out = 0.0;
	int necrosis_model_index = PhysiCell_constants::necrosis_death_model;
	Cell* deadCell;
	// if (deadCell == NULL){std::cout<<"im null "<<std::endl;}
	for( int i=0; i < (*all_cells).size() ; i++ )
	{	
		// std::cout<<"phenotype.cycle.current_phase().code "<<(*all_cells)[i]->phenotype.cycle.current_phase().code<<std::endl;

		

		if( (*all_cells)[i]->phenotype.cycle.current_phase().code == 101 && (*all_cells)[i]->type == 1 )
		//  if(  (*all_cells)[i]->type == 1 )
		{	
			// std::cout<<(*all_cells)[i]->ID<<std::endl;
			 out += 1.0; } 
	}
	// std::cout<<necrosis_model_index<<std::endl;
	// out = necrotic_dead-out;
	return out; 

}
double total_infected_epithelial_cell_count()
{
	int infected_count = 0;
    double infection_threshold = 1.0;  // The threshold value for infection

    // Iterate over all cells
    for( int i = 0; i < (*all_cells).size(); i++ )
    {
        Cell* pCell = (*all_cells)[i];

	if(pCell->type==1){
		// Access the microenvironment variables for the current cell
        double virion = pCell->nearest_density_vector()[microenvironment.find_density_index( "virion" )];
		
        // Check if the variable value exceeds the threshold
        if( virion > infection_threshold )

        {
            std::cout<<virion<<std::endl;
			infected_count++;
        }
	}
	}
	return infected_count;	
}
