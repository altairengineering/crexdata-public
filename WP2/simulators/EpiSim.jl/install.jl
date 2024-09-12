using Pkg

# run this in an interactive session in MN5 or it will time-out !!
# note that you should have installed the dependencies before running this script
Pkg.activate(".")

using PackageCompiler
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--compile", "-c"
            help = "Compile the simulator into a single precompiled excecutable"
            action = :store_true
        "--target", "-t"
            help = "Target folder where the single excecutable will be stored"
            default ="."
    end
    return parse_args(s)
end



args = parse_commandline()
if args["compile"]
    build_folder = "build"
    create_app(pwd(), build_folder, force=true)
    bin_path = abspath(joinpath(build_folder, "bin", "EpiSim"))
    symlink_path = joinpath(args["target"], "episim")
    if !islink(symlink_path)
        symlink(bin_path, symlink_path)
    end
end