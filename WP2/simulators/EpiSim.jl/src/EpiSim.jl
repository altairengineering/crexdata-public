module EpiSim

# using MMCACovid19Vac
using ArgParse
using Dates, Logging, Printf
using HDF5, DataFrames, NetCDF

import JSON
import CSV


include("commands.jl")

function julia_main()::Cint
    """
    This is the entrypoint for the compiled version of EpiSim.
    """
    try
        args = parse_command_line()
        command = args["%COMMAND%"]
        engine = args["engine"]
    
        # Check if the provided command is in the list of accepted commands
        if !(command in COMMANDS)
            println("Unknown command: $command")
            println("Accepted commands are: $(join(COMMANDS, ", "))")
            return 1
        end

        engine = get_engine(engine)

        if command == "run"
            execute_run(args["run"], engine)
        elseif command == "setup"
            execute_setup(args["setup"], engine)
        elseif command == "init"
            execute_init()
        end
        @info "done in main"

    catch e
        @error "error in main" exception=(e, catch_backtrace())
        return 1
    end
    @info "final"
    return 0
end

function main()
    "alias for convenience"
    @info "main"
    try
        return julia_main()
    catch e
        @error "Error after julia_main" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

end # module EpiSim