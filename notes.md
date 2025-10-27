# 21/10/2025

- In the playground you have established that there is definitly a method of inferring causality.

- We have seen this by recreating the example on page 63 of the book.

- However we have only estabished it for when both the Cause and the Effect are uniform and have not established it for a normal nor have we tested it on any other distrabution.

- The reason for this causal inference is likely cus uniform dists have hard boundries which cause this which do not exist in normals. It may be that the inference is based on boundries however and not the distrabution in those boundries, for example I cannot infer whether this method would work with an normal with hard boundries set like a uniform.

# 22/10/2025

- Apprecieate that things take time and a project like this will take time. Be really kind on yourself

- I am not sure what the correct structure is? Should I keep the `fit_{x}_reg` methods?'

- Lets think about what the API, testing and documentation should look like?

```
├── example.json
├── index.html
├── index.js
├── package.json
├── package-lock.json
├── README.md
└── CausalInference
    ├── ModelObjects/WrapperFunction
    ├── models.js
    ├── routes.js
    └── MethodsOfInference
        ├── Combined Or automatic solver combining other methods of correlation
```

What does this model wrapper object look like?

I'm thinking of a lot of stuff but I think that the first thing I should do is get a working example of the correlation inference