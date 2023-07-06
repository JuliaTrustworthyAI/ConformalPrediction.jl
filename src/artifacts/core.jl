using LazyArtifacts
using Serialization

function artifact_folder()
    jversion = "$(Int(VERSION.major)).$(Int(VERSION.minor))"
    candidate_tag = "artifacts-$jversion"
    path = try
        joinpath(@artifact_str(candidate_tag), "$candidate_tag")
    catch
        @error "No artifacts found for Julia version $jversion."
    end
    return path
end
