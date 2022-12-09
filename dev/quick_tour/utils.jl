function multi_slider(vals::Dict; title="")
    
    return PlutoUI.combine() do Child
        
        inputs = [
            md""" $(_name): $(
                Child(_name, Slider(_vals[1], default=_vals[2], show_value=true))
            )"""
            
            for (_name, _vals) in vals
        ]
        
        md"""
        #### $title
        $(inputs)
        """
    end
    
end