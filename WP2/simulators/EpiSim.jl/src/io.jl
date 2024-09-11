function update_config!(config, cmd_line_args)
    # Define dictionary containing epidemic parameters

    # overwrite config with command line
    if cmd_line_args["start-date"] !== nothing
        config["simulation"]["start_date"] = cmd_line_args["start-date"]
    end
    if cmd_line_args["end-date"] !== nothing
        config["simulation"]["end_date"] = cmd_line_args["end-date"]
    end
    if cmd_line_args["export-compartments-time-t"] !== nothing
        config["simulation"]["export_compartments_time_t"] = cmd_line_args["export-compartments-time-t"]
    end
    if cmd_line_args["export-compartments-full"] == true
        config["simulation"]["export_compartments_full"] = true
    end

    nothing
end

abstract type AbstractOutputFormat end

struct NetCDFFormat <: AbstractOutputFormat end
struct HDF5Format <: AbstractOutputFormat end

const OUTPUT_FORMATS = Dict("netcdf" => NetCDFFormat(), "hdf5" => HDF5Format())

get_output_format(output_format::String) = get(OUTPUT_FORMATS, output_format, NetCDFFormat())
get_output_format_str(output_format::AbstractOutputFormat) = findfirst(==(output_format), OUTPUT_FORMATS)

function save_full(epi_params, population, output_path::String, output_format::Union{String,AbstractOutputFormat}; kwargs...)
    format = output_format isa String ? get_output_format(output_format) : output_format
    _save_full(epi_params, population, output_path, format; kwargs...)
end

function _save_full(epi_params, population, output_path::String, ::NetCDFFormat; G_coords=String[], M_coords=String[], T_coords=String[])
    filename = joinpath(output_path, "compartments_full.nc")
    @info "Storing full simulation output in NetCDF: $filename"
    try
        save_simulation_netCDF(epi_params, population, filename; G_coords, M_coords, T_coords)
    catch e
        @error "Error saving simulation output" exception=(e, catch_backtrace())
        rethrow(e)
    end
    @info "done saving ??"
end

function _save_full(epi_params, population, output_path::String, ::HDF5Format; kwargs...)
    filename = joinpath(output_path, "compartments_full.h5")
    @info "Storing full simulation output in HDF5: $filename"
    save_simulation_hdf5(epi_params, population, filename)
end

function save_time_step(epi_params, population, output_path::String, export_compartments_time_t::Int) 
    export_compartments_date = first_day + Day(export_compartments_time_t - 1)
    filename = joinpath(output_path, "compartments_t_$(export_compartments_date).h5")
    @info "Storing compartments at single date $(export_compartments_date):"
    @info "\t- Simulation step: $(export_compartments_time_t)"
    @info "\t- filename: $(filename)"
    save_simulation_hdf5(epi_params, population, filename; 
                        export_time_t = export_compartments_time_t)
end