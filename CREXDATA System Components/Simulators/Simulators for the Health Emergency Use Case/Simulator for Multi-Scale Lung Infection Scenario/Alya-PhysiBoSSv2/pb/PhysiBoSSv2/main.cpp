/*
###############################################################################
# If you use PhysiCell in your project, please cite PhysiCell and the version #
# number, such as below:                                                      #
#                                                                             #
# We implemented and solved the model using PhysiCell (Version x.y.z) [1].    #
#                                                                             #
# [1] A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin, #
#     PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellu-  #
#     lar Systems, PLoS Comput. Biol. 14(2): e1005991, 2018                   #
#     DOI: 10.1371/journal.pcbi.1005991                                       #
#                                                                             #
# See VERSION.txt or call get_PhysiCell_version() to get the current version  #
#     x.y.z. Call display_citations() to get detailed information on all cite-#
#     able software used in your PhysiCell application.                       #
#                                                                             #
# Because PhysiCell extensively uses BioFVM, we suggest you also cite BioFVM  #
#     as below:                                                               #
#                                                                             #
# We implemented and solved the model using PhysiCell (Version x.y.z) [1],    #
# with BioFVM [2] to solve the transport equations.                           #
#                                                                             #
# [1] A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin, #
#     PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellu-  #
#     lar Systems, PLoS Comput. Biol. 14(2): e1005991, 2018                   #
#     DOI: 10.1371/journal.pcbi.1005991                                       #
#                                                                             #
# [2] A Ghaffarizadeh, SH Friedman, and P Macklin, BioFVM: an efficient para- #
#     llelized diffusive transport solver for 3-D biological simulations,     #
#     Bioinformatics 32(8): 1256-8, 2016. DOI: 10.1093/bioinformatics/btv730  #
#                                                                             #
###############################################################################
#                                                                             #
# BSD 3-Clause License (see https://opensource.org/licenses/BSD-3-Clause)     #
#                                                                             #
# Copyright (c) 2015-2018, Paul Macklin and the PhysiCell Project             #
# All rights reserved.                                                        #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions are met: #
#                                                                             #
# 1. Redistributions of source code must retain the above copyright notice,   #
# this list of conditions and the following disclaimer.                       #
#                                                                             #
# 2. Redistributions in binary form must reproduce the above copyright        #
# notice, this list of conditions and the following disclaimer in the         #
# documentation and/or other materials provided with the distribution.        #
#                                                                             #
# 3. Neither the name of the copyright holder nor the names of its            #
# contributors may be used to endorse or promote products derived from this   #
# software without specific prior written permission.                         #
#                                                                             #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" #
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE   #
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE  #
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE   #
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR         #
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF        #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS    #
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN     #
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)     #
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE  #
# POSSIBILITY OF SUCH DAMAGE.                                                 #
#                                                                             #
###############################################################################
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cmath>
#include <omp.h>
#include <fstream>
#include <algorithm>    // std::rotate
#include <filesystem>  
#include <thread>      
#include <chrono>      
#include "./core/PhysiCell.h"
#include "./modules/PhysiCell_standard_modules.h" 
#include "./addons/PhysiBoSS/src/maboss_intracellular.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>
// put custom code modules here! 

#include "./custom_modules/custom.h" 
	
using namespace BioFVM;
using namespace PhysiCell;



std::string COVID19_version = "0.6.0"; 

double DCAMOUNT = 0;
double DM = 0; // global ICs
double DL = 0; // global ICs
double TC = 0;
double TH1 = 0;
double TH2 = 0;
double TCt = 0;
double Tht = 0;
double Bc = 0;
double Ps = 0;
double Ig = 0;
double TCN = 0;
double THN = 0;
double BN = 0;
double EPICOUNT = 1;
double tissueCD4=0;
double tissueCD8=0;

std::vector<int> history(144001); //144000 - full day delay - set max (lets say a day) delay, let user define up to that amt delay.
std::vector<int> historyTc(121); //120 - half day delay
std::vector<int> historyTh(121);
//size 72000 - 0.5 day -> 0.01min
void checkKillFileExistsAndExit(const std::string& folderPath) {
    std::string filePath = folderPath + "/kill.txt";
    std::ifstream file(filePath.c_str());
    if (file.good()) {
        std::cout << "kill.txt file found. Exiting program." << std::endl;
        exit(EXIT_SUCCESS);
    }
	else{
		std::cout<<"no kill found in "<< folderPath<< std::endl;
	}
}
// when in marenostrum this is done through files else its done through kafka messsages
void checkChangeFileExists(const std::string& folderPath) {
    std::string filePath = folderPath + "/changes.txt";
    std::ifstream file(filePath.c_str());
    if (file.good()) {
		std::cout << "changes.txt file found. Proceeding with changes." << std::endl;
		// Read the contents of the file
		// introduce the knock out to the boolean model.
		// for example FADD
		
    }
	else{
		std::cout<<"no changes are introduced "<< folderPath<< std::endl;
	}
}

// Helper function to recursively create directories (like `mkdir -p`)
bool makeDirectoryRecursively(const std::string& path) {
    if (path.empty()) return false;

    size_t pos = 0;
    do {
        pos = path.find_first_of('/', pos + 1);
        std::string subdir = path.substr(0, pos);

        if (subdir.empty()) continue;

        struct stat st;
        if (stat(subdir.c_str(), &st) != 0) {
            if (mkdir(subdir.c_str(), 0755) != 0 && errno != EEXIST) {
                std::cerr << "mkdir failed: " << std::strerror(errno) << std::endl;
                return false;
            }
        } else if (!S_ISDIR(st.st_mode)) {
            std::cerr << subdir << " exists but is not a directory." << std::endl;
            return false;
        }
    } while (pos != std::string::npos);

    return true;
}

bool ensureDirectoryExists(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    } else {
        return makeDirectoryRecursively(path);
    }
}

bool checkroundfileexistswait(const std::string& folderPath) {
    std::string filePath = folderPath + "/round.txt";
	std::cout<<"checking for round.txt in "<<folderPath<<std::endl;
    while (true) {
        std::ifstream file(filePath.c_str());
        if (file.good()) {
            std::cout << "round.txt file found. Proceeding with next round if it contains 0 else quit." << std::endl;
            std::string line;
            if (std::getline(file, line)) {
                if (line == "0") {
                    return 1;
                }
				else if (line == "1") {
					std::cout << "round.txt file contains tag 1. Exiting program." << std::endl;
					exit(EXIT_SUCCESS);
				}
            }
        } 

		else {
            std::cout << "round.txt file not found. looking... "<<folderPath << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(40));  
        }
    }
}
std::string get_parent_path(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos != std::string::npos)
        return path.substr(0, pos);
    return path;
}
int main( int argc, char* argv[] )
{
	// load and parse settings file(s)
	
	bool XML_status = false; 
	if( argc > 1 )
	{ XML_status = load_PhysiCell_config_file( argv[1] ); }
	else
	{ XML_status = load_PhysiCell_config_file( "./config/PhysiCell_settings.xml" ); }
	if( !XML_status )
	{ exit(-1); }

	//Define initial values for Globals
	DM = parameters.doubles("DM_init");
	DL = parameters.doubles("DL_init");
	TC = parameters.doubles("TC_init");
	TH1 = parameters.doubles("TH1_init");
	TH2 = parameters.doubles("TH2_init");
	TCt = parameters.doubles("TCt_init");
	Tht = parameters.doubles("Tht_init");
	Bc = parameters.doubles("Bc_init");
	Ps = parameters.doubles("Ps_init");
	Ig = parameters.doubles("Ig_init");
	TCN = parameters.doubles("TCN_init");
	THN= parameters.doubles("THN_init");
	BN = parameters.doubles("BN_init");
	

	int rounds = parameters.ints("rounds");
	int current_round=0;
	// so each simualtion should follow the following
	// xml max time is the max time of a single round
	double round_time = PhysiCell_settings.max_time;
	// PhysiCell_settings.max_time= PhysiCell_settings.max_time*rounds;
	// create round directory

	// OpenMP setup
	omp_set_num_threads(PhysiCell_settings.omp_num_threads);
	
	// time setup 
	std::string time_units = "min"; 

	/* Microenvironment setup */ 
	
	setup_microenvironment(); // modify this in the custom code 
	
	/* PhysiCell setup */ 
 	
	// set mechanics voxel size, and match the data structure to BioFVM
	// If this is not provided in the .xml, it will return 0 which leads to a memory explosion and crash
	double mechanics_voxel_size = parameters.doubles("mech_voxel_size"); 
    if (mechanics_voxel_size < 1.e-6)
    {
        std::cout << "ERROR: mechanics_voxel_size (not a user_param in .xml?)= " << mechanics_voxel_size << std::endl;
        std::cout << "* probably not provided as a user_param in .xml " << std::endl;
        std::cout << "* setting = 30" << std::endl;
        // exit(-1);
	    mechanics_voxel_size = 30.0;
    }
	Cell_Container* cell_container = create_cell_container_for_microenvironment( microenvironment, mechanics_voxel_size );
	create_cell_types();
	setup_tissue();
	set_save_biofvm_mesh_as_matlab( true ); 
	set_save_biofvm_data_as_matlab( true ); 
	set_save_biofvm_cell_data( true ); 
	set_save_biofvm_cell_data_as_custom_matlab( true );
	std::string initial_folder =PhysiCell_settings.folder;
	// save a simulation snapshot 
	while(current_round<=rounds){
		std::cout << "Saving simulation snapshot for round " << current_round << std::endl;
		// set current folder of round
		PhysiCell_settings.folder = initial_folder + "round_" + std::to_string(current_round);
		// create folder if it doesnt exist
		ensureDirectoryExists(PhysiCell_settings.folder);



		// initial foplder = runs/sim_1/output/
    // Get the parent path
    	std::string simulation_folder = get_parent_path(initial_folder);
		std::string turbine = get_parent_path(simulation_folder);
		std::string round_folder = turbine+"/round_" + std::to_string(current_round);
		std::cout<<" PhysiCell_settings.folder "<<PhysiCell_settings.folder<<std::endl;
		std::cout<<" initial_folder "<<initial_folder<<std::endl;
		std::cout<<" simulation_folder "<<simulation_folder<<std::endl;
		std::cout<<" turbine "<<turbine<<std::endl;
		std::cout<<" round_folder "<<round_folder<<std::endl;
		

		//round_folder =  runs/sim_1/round_1
		// we want to look for the round.txt with the runs/round_i 
		// if round not 0 you need to wait to start new round
		if(current_round!=0){
			std::string path = "" + std::to_string(current_round);
			checkroundfileexistswait(round_folder);
			checkKillFileExistsAndExit(PhysiCell_settings.folder);
			std::string moi_file=simulation_folder+"/moi.txt";
			std::cout<<"moi file "<<moi_file<<std::endl;			
			std::ifstream file(moi_file);
			std::stringstream buffer;
			buffer << file.rdbuf();
			std::string contents = buffer.str();
			parameters.doubles("multiplicity_of_infection") =  std::stod(contents);
			parameters.strings("input_virion_alya") = round_folder+"/input_virion_alya.txt";
			std::string alya_voxel_file = round_folder+"/input_virion_alya.txt";
			load_alya_infected_voxels(alya_voxel_file);
    		set_virion_concentration();
    		save_alya_infected_voxels(alya_voxel_file);
			PhysiCell_globals.full_output_index = 0;
		}
	// initialize
	// start round loop here
	char filename[1024];
	sprintf( filename , "%s/initial" , PhysiCell_settings.folder.c_str() ); 
	save_PhysiCell_to_MultiCellDS_xml_pugi( filename , microenvironment , PhysiCell_globals.current_time ); 
	
	sprintf( filename , "%s/states_initial.csv", PhysiCell_settings.folder.c_str());
	MaBoSSIntracellular::save( filename, *PhysiCell::all_cells);
	
	// save a quick SVG cross section through z = 0, after setting its 
	// length bar to 200 microns 

	PhysiCell_SVG_options.length_bar = 200; 

	// for simplicity, set a pathology coloring function 
	
	std::vector<std::string> (*cell_coloring_function)(Cell*) = tissue_coloring_function; 
	
	sprintf( filename , "%s/initial.svg" , PhysiCell_settings.folder.c_str() ); 
				std::cout << "\n===============================" << std::endl;
				std::cout << "Starting round " << current_round << " of " << rounds << std::endl;
				std::cout << "===============================\n" << std::endl;
				std::cout << "Saving simulation snapshot for round " << current_round << std::endl;
	
	sprintf( filename , "%s/legend.svg" , PhysiCell_settings.folder.c_str() ); 
	create_plot_legend( filename , cell_coloring_function ); 
	
	display_citations(); 
	
	// set the performance timers 

	BioFVM::RUNTIME_TIC();
	BioFVM::TIC();
	
	std::ofstream report_file;
	// char filename_rf[1024];

	sprintf( filename , "%s/simulation_report.txt" , PhysiCell_settings.folder.c_str() ); 
	report_file.open(filename);
	report_file<<"process_id\ttimepoint\tnum_all_cells\tnum_total_epithelial\tnum_alive_epithelial\tnum_apoptotic_epithelial\tnum_necrotic_epithelial\tnum_infected_epithelial"<<std::endl;
	// report_file.close();

    
					std::cout << "\n--- Waiting for round.txt for round " << current_round << " ---" << std::endl;
	std::string path = "" + std::to_string(current_round);
	std::ofstream dm_tc_file;
	sprintf( filename , "%s/dm_tc.dat" , PhysiCell_settings.folder.c_str() ); 
	dm_tc_file.open (filename);
	std::cout<<"starting rounds"<<std::endl;
	double max_sim_time = (PhysiCell_settings.max_time + 0.1*diffusion_dt)/3;
	std::cout<<"maxsim time "<<max_sim_time*(current_round+1) <<std::endl;
	try 
	{	
		std::cout<<"maxsim time "<<max_sim_time <<std::endl;	
		while( PhysiCell_globals.current_time < max_sim_time*(current_round+1) )
		{
			// save data if it's time. 
			if( fabs( PhysiCell_globals.current_time - PhysiCell_globals.next_full_save_time ) < 0.01 * diffusion_dt )
			{
				display_simulation_status( std::cout ); 
    			checkKillFileExistsAndExit(PhysiCell_settings.folder);

				report_file << PhysiCell_settings.folder.c_str() <<","
				<< PhysiCell_globals.current_time << ","
				<< total_cell_count() << ","
				<< total_epithelial_cell_count() << ","
				<< total_alive_epithelial_cell_count() << ","
				<< total_apoptotic_epithelial_cell_count() << ","
				<< total_necrotic_epithelial_cell_count() << ","
				<< total_infected_epithelial_cell_count()
				<< std::endl;
				// report_file.close();
				
				if( PhysiCell_settings.enable_full_saves == true )
					
				{	
					sprintf( filename , "%s/output%08u" , PhysiCell_settings.folder.c_str(),  PhysiCell_globals.full_output_index ); 
					
					dm_tc_file << DM << " " << TC << " " << TH1 << " " << TH2 << " " << TCt << " " << Tht <<" " << Bc <<" " << Ps <<" " << Ig <<" " << TCN <<" " << THN <<" " << BN <<" " << DL <<std::endl; //write globals data
					
					save_PhysiCell_to_MultiCellDS_xml_pugi( filename , microenvironment , PhysiCell_globals.current_time ); 

					// sprintf( filename , "%s/states_%08u.csv", PhysiCell_settings.folder.c_str(), PhysiCell_globals.full_output_index);
					// MaBoSSIntracellular::save( filename, *PhysiCell::all_cells );
				}
				
				PhysiCell_globals.full_output_index++; 
				PhysiCell_globals.next_full_save_time += PhysiCell_settings.full_save_interval;
			}
			
			// save SVG plot if it's time
			if( fabs( PhysiCell_globals.current_time - PhysiCell_globals.next_SVG_save_time  ) < 0.01 * diffusion_dt )
			{
				if( PhysiCell_settings.enable_SVG_saves == true )
				{	
					sprintf( filename , "%s/snapshot%08u.svg" , PhysiCell_settings.folder.c_str() , PhysiCell_globals.SVG_output_index ); 
					SVG_plot( filename , microenvironment, 0.0 , PhysiCell_globals.current_time, cell_coloring_function );
					
					PhysiCell_globals.SVG_output_index++; 
					PhysiCell_globals.next_SVG_save_time  += PhysiCell_settings.SVG_save_interval;
				}
			}

			// update the microenvironment
			microenvironment.simulate_diffusion_decay( diffusion_dt );
            
			// history functions		
			DC_history_main_model( diffusion_dt );
			
			//external_immune_main_model( diffusion_dt );
			external_immune_model( diffusion_dt );
			
			// receptor dynamics 			
			receptor_dynamics_main_model( diffusion_dt );
			
			// detach dead cells 
			// detach_all_dead_cells( diffusion_dt );
			
			cells_to_move_from_edge.clear();
			
			// run PhysiCell 
			((Cell_Container *)microenvironment.agent_container)->update_all_cells( PhysiCell_globals.current_time );
			
			/*
			  Custom add-ons could potentially go here. 
			*/
            process_tagged_cells_on_edge();
			if( fabs(remainder(PhysiCell_globals.current_time, phenotype_dt)) < 0.01 * diffusion_dt )
			{
				immune_cell_recruitment( phenotype_dt );
			}

			PhysiCell_globals.current_time += diffusion_dt;
		}
		
		if( PhysiCell_settings.enable_legacy_saves == true )
		{			
			log_output(PhysiCell_globals.current_time, PhysiCell_globals.full_output_index, microenvironment, report_file);
			report_file.close();
		}
	}
	catch( const std::exception& e )
	{ // reference to the base of a polymorphic object
		std::cout << e.what(); // information from length_error printed
	}
	
	// save a final simulation snapshot 
	
	sprintf( filename , "%s/final" , PhysiCell_settings.folder.c_str() ); 
	save_PhysiCell_to_MultiCellDS_xml_pugi( filename , microenvironment , PhysiCell_globals.current_time ); 

	sprintf( filename , "%s/final.csv", PhysiCell_settings.folder.c_str());
	MaBoSSIntracellular::save( filename, *PhysiCell::all_cells );
	
		
	sprintf( filename , "%s/final.svg" , PhysiCell_settings.folder.c_str() ); 
	// SVG_plot( filename , microenvironment, 0.0 , PhysiCell_globals.current_time, cell_coloring_function );
	SVG_plot_virus( filename , microenvironment, 0.0 , PhysiCell_globals.current_time, cell_coloring_function );

	
	// timer 
	
	std::cout << std::endl << "Total simulation runtime of round: "<<current_round << std::endl; 
	BioFVM::display_stopwatch_value( std::cout , BioFVM::runtime_stopwatch_value() ); 
	
	extern int recruited_neutrophils; 
	extern int recruited_Tcells; 
	extern int recruited_macrophages; 
	
	extern double first_macrophage_recruitment_time;
	extern double first_neutrophil_recruitment_time; 
	extern double first_CD8_T_cell_recruitment_time; 

	std::cout << std::endl; 
	std::cout << "recruited macrophages: " << recruited_macrophages << " starting at time " 
		<< first_macrophage_recruitment_time <<	std::endl; 
	std::cout << "recruited neutrophils: " << recruited_neutrophils << " starting at time " 
		<< first_neutrophil_recruitment_time << std::endl; 
	std::cout << "recruited T cells: " << recruited_Tcells << " starting at time "
		<< first_CD8_T_cell_recruitment_time << std::endl << std::endl; 	
	// recruited_neutrophils = 0; 
	// recruited_Tcells = 0; 
	// recruited_macrophages = 0; 

	// for( int n =0 ; n < (*all_cells).size() ; n++ )
	// {
	// 	Cell* pC = (*all_cells)[n]; 
	// 	if( pC->type == 5 )
	// 	{ recruited_neutrophils++; }
	// 	if( pC->type == 3 )
	// 	{ recruited_Tcells++; }
	// 	if( pC->type == 4 )
	// 	{ recruited_macrophages++; }
	// }
	// std::cout << "remaining macrophages: " << recruited_macrophages << std::endl; 
	// std::cout << "remaining neutrophils: " << recruited_neutrophils << std::endl; 
	// std::cout << "remaining T cells: " << recruited_Tcells << std::endl; 
	current_round++;


	}



	
	

		

	


	return 0; 
}
