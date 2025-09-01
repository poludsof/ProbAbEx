function load_celeba_images(root::AbstractString; out_size=(64,64), limit::Union{Nothing,Int}=nothing, shuffle::Bool=true)
    
    files = filter(f -> endswith(lowercase(f), ".jpg") || endswith(lowercase(f), ".png"),
                   readdir(root; join=true))
                   
    shuffle && Random.shuffle!(files)

    limit !== nothing && (files = files[1:limit])

    H, W = out_size; C = 3; D = H*W*C
    X = Array{Float32}(undef, D, length(files))

    for (i, fp) in enumerate(files)
        img = load(fp)                  # could be H×W Colorant, or H×W×C numeric
        img = RGB.(img)                 # ensure 2-D Colorant array H×W

        # center crop to square on the *first two* dims
        H0, W0 = size(img, 1), size(img, 2)
        s = min(H0, W0)
        top  = (H0 - s) ÷ 2 + 1
        left = (W0 - s) ÷ 2 + 1
        cropped = @view img[top:top+s-1, left:left+s-1]   # <-- no third-dim index

        # resize the 2-D Colorant array
        img_res = ImageTransformations.imresize(cropped, (H, W))  # H×W Colorant

        # to H×W×C Float32 then flatten to D
        HWC = Float32.(permutedims(channelview(img_res), (2,3,1)))  # H×W×C, Float32 in [0,1]
        X[:, i] = reshape(HWC, D)
    end
    return X
end

# Each column of data is one image, flattened, with values in [0, 1]
data = load_celeba_images("/home/poludsof/ProbAbEx/src/datasets/img_align_celeba", out_size=(32, 32), limit=1000, shuffle=true)
data[1]

H, W, C = 32, 32, 3
k = 6

size(data)

show_image_vec(data[:, k]; H=H, W=W, title="CelebA image $k", savepath="celeba_img_$k.png")
show_image90(data[:, k]; H=H, W=W)

show_image90(x; H::Int, W::Int) = begin
    img = colorview(RGB{Float32},
                    permutedims(reshape(Float32.(x), H, W, 3), (3,1,2)))
    display(img)
end

function show_image_vec(x::AbstractVector; H::Int, W::Int, title::AbstractString="Image", savepath::Union{Nothing,String}=nothing)
    @assert length(x) == H*W*3
    img3 = reshape(Float32.(x), H, W, 3)          # H×W×3
    ch  = permutedims(img3, (3, 1, 2))            # 3×H×W (channels-first)
    img = colorview(RGB{Float32}, ch)             # H×W Array{RGB{Float32}}
    img = rotr90(img)

    fig = Figure(resolution=(W*8, H*8), fontsize=14)

    ax  = Axis(fig[1,1], title=title)

    image!(ax, img, interpolate=false) 
    hidespines!(ax); hidedecorations!(ax, grid=false)

    if savepath !== nothing
        save(savepath, fig)
    end

    display(rotr90(img))
    return fig
end