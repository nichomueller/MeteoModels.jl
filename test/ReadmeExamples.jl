module ReadmeExamplesTest

using Opal
using Test

function readme_blocks(readme_path)
  text = read(readme_path,String)
  blocks = String[]
  for m in eachmatch(r"```julia\n(.*?)```"s,text)
    preceding = text[max(firstindex(text),m.offset-60):m.offset]
    occursin("readme-test:skip",preceding) && continue
    push!(blocks,m.captures[1])
  end
  blocks
end

@testset "README examples" begin
  readme_path = joinpath(pkgdir(Opal),"README.md")
  blocks = readme_blocks(readme_path)
  @test !isempty(blocks)

  mktempdir() do dir
    script = joinpath(dir,"readme_examples.jl")
    write(script,join(blocks,"\n"))
    sandbox = Module(:ReadmeSandbox)
    Core.eval(sandbox,:(using Opal))
    @test (Base.include(sandbox,script); true)
  end
end

end # module
