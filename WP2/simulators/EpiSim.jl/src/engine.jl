using NetCDF
using NCDatasets
using DataFrames
using MMCACovid19Vac

include("io.jl")

const ENGINES  = ["MMCACovid19Vac", "MMCACovid19"]
const COMMANDS = ["run", "setup", "init"]

abstract type AbstractEngine end

# Add to this as we add more engines
struct MMCACovid19VacEngine <: AbstractEngine end
struct MMCACovid19Engine <: AbstractEngine end

# Define a dictionary to map engine names to their types
const ENGINE_TYPES = Dict(
    "MMCACovid19Vac" => MMCACovid19VacEngine,
    "MMCACovid19" => MMCACovid19Engine
)

function get_engine(engine_name::String)
    engine_type = get(ENGINE_TYPES, engine_name, nothing)
    isnothing(engine_type) && error("Unknown engine: $engine_name")
    return engine_type()
end

function validate_config(config, ::MMCACovid19VacEngine)
    @assert haskey(config, "simulation")
    @assert haskey(config, "data")
    @assert haskey(config, "epidemic_params")
    @assert haskey(config, "population_params")
    @assert haskey(config, "vaccination")
    @assert haskey(config, "NPI")
end

function validate_config(config, ::MMCACovid19Engine)
    @assert haskey(config, "simulation")
    @assert haskey(config, "data")
    @assert haskey(config, "epidemic_params")
    @assert haskey(config, "population_params")
    @assert haskey(config, "NPI") 
end


function read_input_files(::AbstractEngine, config::Dict, data_path::String, instance_path::String, init_condition_path::String)
    data_dict       = config["data"]
    simulation_dict = config["simulation"]
    npi_params_dict = config["NPI"]
    init_format      = get(simulation_dict, "init_format", "netcdf")

    #########################
    # Simulation output 
    #########################
    output_path = joinpath(instance_path, "output")
    if !isdir(output_path)
        println("Creating output folder: $output_path")
        mkpath(output_path)
    end


    #########################################################
    # Containment measures
    #########################################################

    # Daily Mobility reduction
    kappa0_filename = get(data_dict, "kappa0_filename", nothing)
    first_day = Date(simulation_dict["start_date"])
    npi_params = init_NPI_parameters_struct(data_path, npi_params_dict, kappa0_filename, first_day)
    # vac_parms = Vaccination_Params(tᵛs, ϵᵍs)


    #####################
    # Initial Condition
    #####################

    if !isfile(init_condition_path) || length(init_condition_path) == 0
        init_condition_path = joinpath(data_path, get(data_dict, "initial_condition_filename", nothing))
    end

    # use initial compartments matrix to initialize simulations
    if init_format == "netcdf"
        @info "Reading initial conditions from: $(init_condition_path)"
        initial_compartments = ncread(init_condition_path, "data")
    elseif init_format == "hdf5"
        # TODO: does this path work?
        initial_compartments = h5open(init_condition_path, "r") do file
            read(file, "data")
        end
    else
        @error "init_format must be one of : netcdf/hdf5"
        return 1
    end

    # Loading mobility network
    mobility_matrix_filename = joinpath(data_path, data_dict["mobility_matrix_filename"])
    network_df  = CSV.read(mobility_matrix_filename, DataFrame)

    # Loading metapopulation patches info (surface, label, population by age)
    metapop_data_filename = joinpath(data_path, data_dict["metapopulation_data_filename"])
    metapop_df = CSV.read(metapop_data_filename, DataFrame, types=Dict(:id => String))

    return npi_params, network_df, metapop_df, initial_compartments
end

"""
Run the engine using input files (which must be available in the data_path and instance_path)
and save the output to the output folder.
"""
function run_engine_io(engine::AbstractEngine, config::Dict, data_path::String, instance_path::String, init_condition_path::String)
    simulation_dict = config["simulation"]
    output_format    = simulation_dict["output_format"]
    save_full_output = get(simulation_dict, "save_full_output", false)
    time_step_tosave   = get(simulation_dict, "export_compartments_time_t", nothing)
    output_path = joinpath(instance_path, "output")

    # if output_path does not exist, create it
    if !isdir(output_path)
        mkpath(output_path)
    end

    npi_params, network_df, metapop_df, initial_compartments = read_input_files(engine, config, data_path, instance_path, init_condition_path)

    epi_params, population, coords = run_engine(engine, config, npi_params, network_df, metapop_df, initial_compartments)

    @info "\t- Save full output = " save_full_output
    if time_step_tosave !== nothing
        @info "\t- Save time step at t=" time_step_tosave
        @assert time_step_tosave isa Int
    end

    if save_full_output
        save_full(epi_params, population, output_path, output_format; coords...)
    end
    if time_step_tosave !== nothing
        save_time_step(epi_params, population, output_path, time_step_tosave)
    end

    @info "done running engine io"
end

"""
Run the engine using Julia data structures as inputs. Does not save the output to file.

TODO: decouple from MMCACovid19Vac.jl (NPI_Params)
"""
function run_engine(::MMCACovid19VacEngine, config::Dict, npi_params::NPI_Params, network_df::DataFrame, metapop_df::DataFrame, initial_compartments::Array{Float64, 4})
    @info "Running MMCACovid19VacEngine"
    simulation_dict = config["simulation"]
    epi_params_dict = config["epidemic_params"]
    pop_params_dict = config["population_params"]
    vac_params_dict = config["vaccination"]



    ########################################
    ####### VARIABLES INITIALIZATION #######
    ########################################
    @info "Initializing variables"

    # Reading simulation start and end dates
    first_day = Date(simulation_dict["start_date"])
    last_day  = Date(simulation_dict["end_date"])
    # Converting dates to time steps
    T = (last_day - first_day).value + 1
    # Array with time coordinates (dates)
    T_coords  = string.(collect(first_day:last_day))

    # Metapopulations patches coordinates (labels)
    M_coords = map(String,metapop_df[:, "id"])
    M = length(M_coords)

    # Coordinates for each age strata (labels)
    G_coords = map(String, pop_params_dict["G_labels"])
    G = length(G_coords)

    ####################################################
    #####   INITIALIZATION OF DATA Structures   ########
    ####################################################
    @info "Initializing data structures"

    ## POPULATION PARAMETERS
    population = init_pop_param_struct(G, M, G_coords, pop_params_dict, metapop_df, network_df)
    ## EPIDEMIC PARAMETERS 
    epi_params = init_epi_parameters_struct(G, M, T, G_coords, epi_params_dict)

    @assert size(initial_compartments) == (G, M, epi_params.V, epi_params.NumComps)


    #########################################################
    # Vaccination parameters
    #########################################################
    @info "Initializing vaccination parameters"

    # vaccionation dates
    start_vacc = vac_params_dict["start_vacc"]
    dur_vacc   = vac_params_dict["dur_vacc"]
    end_vacc   = start_vacc + dur_vacc

    # total vaccinations per age strata
    total_population = sum(population.nᵢᵍ)
    ϵᵍ = vac_params_dict["ϵᵍ"] * round( total_population * vac_params_dict["percentage_of_vacc_per_day"] )
    tᵛs = [start_vacc, end_vacc, T]
    ϵᵍs = ϵᵍ .* [0  Int(vac_params_dict["are_there_vaccines"])  0] 

    ##################################################

    @info "- Initializing MMCA epidemic simulations"
    @info "\t- first_day_simulation = "  first_day
    @info "\t- last_day_simulation = " last_day
    @info "\t- G (agent class) = " G
    @info "\t- M (n. of metapopulations) = "  M
    @info "\t- T (simulation steps) = " T
    @info "\t- V (vaccination states) = " epi_params.V
    @info "\t- N. of epi compartments = " epi_params.NumComps

    ########################################################
    ################ RUN THE SIMULATION ####################
    ########################################################

    set_compartments!(epi_params, population, initial_compartments)

    run_epidemic_spreading_mmca!(epi_params, population, npi_params, tᵛs, ϵᵍs; verbose = true )

    return epi_params, population, Dict(:T_coords => T_coords, :G_coords => G_coords, :M_coords => M_coords)
end


function run_engine(::MMCACovid19Engine, config::Dict, data_path::String, instance_path::String, init_condition_path::String)

    ###########################################
    ############# FILE READING ################
    ###########################################
    
    simulation_dict = config["simulation"]
    data_dict       = config["data"]
    epi_params_dict = config["epidemic_params"]
    pop_params_dict = config["population_params"]
    npi_params_dict = config["NPI"]

    #########################
    # Simulation output 
    #########################
    output_path = joinpath(instance_path, "output")
    if !isdir(output_path)
        println("Creating output folder: $output_path")
        mkpath(output_path)
    end

    output_format    = simulation_dict["output_format"]
    save_full_output = get(simulation_dict, "save_full_output", false)
    save_time_step   = get(simulation_dict, "save_time_step", nothing)
    init_format      = get(simulation_dict, "init_format", "netcdf")

    #####################
    # Initial Condition
    #####################

    if !isfile(init_condition_path) || length(init_condition_path) == 0
        init_condition_path = joinpath(data_path, get(data_dict, "initial_condition_filename", nothing))
    end


    # use initial compartments matrix to initialize simulations
    if init_format == "netcdf"
        @info "Reading initial conditions from: $(init_condition_path)"
        initial_compartments = ncread(init_condition_path, "data")
    elseif init_format == "hdf5"
        initial_compartments = h5open(init_condition_path, "r") do file
            read(file, "data")
        end
    else
        @error "init_format must be one of : netcdf/hdf5"
        return 1
    end


    ########################################
    ####### VARIABLES INITIALIZATION #######
    ########################################

    # Reading simulation start and end dates
    first_day = Date(simulation_dict["start_date"])
    last_day  = Date(simulation_dict["end_date"])
    # Converting dates to time steps
    T = (last_day - first_day).value + 1
    # Array with time coordinates (dates)
    T_coords  = string.(collect(first_day:last_day))

    # Loading metapopulation patches info (surface, label, population by age)
    metapop_data_filename = joinpath(data_path, data_dict["metapopulation_data_filename"])
    metapop_df = CSV.read(metapop_data_filename, DataFrame, types=Dict(:id => String))

    # Loading mobility network
    mobility_matrix_filename = joinpath(data_path, data_dict["mobility_matrix_filename"])
    network_df  = CSV.read(mobility_matrix_filename, DataFrame)

    # Metapopulations patches coordinates (labels)
    M_coords = map(String,metapop_df[:, "id"])
    M = length(M_coords)

    # Coordinates for each age strata (labels)
    G_coords = map(String, pop_params_dict["G_labels"])
    G = length(G_coords)

    ####################################################
    #####   INITIALIZATION OF DATA Structures   ########
    ####################################################

    ## POPULATION PARAMETERS
    # population = MMCACovid19Vac.init_pop_param_struct(G, M, G_coords, pop_params_dict, metapop_df, network_df)
    #  crear estructura a a mano
    ## EPIDEMIC PARAMETERS 
    # epi_params = MMCACovid19Vac.init_epi_parameters_struct(G, M, T, G_coords, epi_params_dict)
    #  crear estructura a a mano

    # @assert size(initial_compartments) == (G, M, epi_params.V, epi_params.NumComps)

    #########################################################
    # Containment measures
    #########################################################

    # Daily Mobility reduction
    kappa0_filename = get(data_dict, "kappa0_filename", nothing)
    npi_params = MMCACovid19Vac.init_NPI_parameters_struct(data_path, npi_params_dict, kappa0_filename, first_day)

    # vac_parms = Vaccination_Params(tᵛs, ϵᵍs)

    ##################################################

    @info "- Initializing MMCA epidemic simulations"
    @info "\t- first_day_simulation = "  first_day
    @info "\t- last_day_simulation = " last_day
    @info "\t- G (agent class) = " G
    @info "\t- M (n. of metapopulations) = "  M
    @info "\t- T (simulation steps) = " T
    @info "\t- N. of epi compartments = " epi_params.NumComps
    @info "\t- Save full output = " save_full_output
    if save_time_step !== nothing
        @info "\t- Save time step at t=" save_time_step
    end

    ########################################################
    ################ RUN THE SIMULATION ####################
    ########################################################

    MMCACovid19Vac.set_compartments!(epi_params, population, initial_compartments)

    MMCACovid19Vac.run_epidemic_spreading_mmca!(epi_params, population, npi_params, tᵛs, ϵᵍs; verbose = true )

end
