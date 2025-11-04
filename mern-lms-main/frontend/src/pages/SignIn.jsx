import { Label, Spinner, TextInput } from "flowbite-react";
import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import {
  signInStart,
  signInSuccess,
  signInFailure,
} from "../redux/user/userSlice";
import LMS from "../assets/LMS.png";
import { Alert, Button } from "@mui/material";

export default function SignIn() {
  const [formData, setFormData] = useState({});
  const { loading, error: errorMessage } = useSelector((state) => state.user);
  const dispatch = useDispatch();
  const navigate = useNavigate();

  useEffect(() => {
    dispatch(signInFailure(null));
  }, [dispatch]);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.id]: e.target.value.trim() });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.email || !formData.password) {
      return dispatch(signInFailure("Please fill out all fields."));
    }
    try {
      dispatch(signInStart());
      const res = await fetch("/api/auth/signin", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      const data = await res.json();
      if (data.success === false) {
        dispatch(signInFailure(data.message));
      }
      if (res.ok) {
        dispatch(signInSuccess(data));
        navigate("/");
      }
    } catch (error) {
      dispatch(signInFailure(error.message));
    }
  };

  return (
    <div className="flex max-w-3xl mx-auto flex-col md:flex-row md:items-center gap-5 my-8">
      <div className="flex-1">
        <Link to={"/"}>
          <img src={LMS} alt="logo" className="w-[340px]" />
        </Link>
        <p className="text-sm mt-4">
          Sign with your account provided by system administrator.
        </p>
      </div>

      <div className="flex-1">
        <form className="flex flex-col gap-4" onSubmit={handleSubmit}>
          <div className="flex flex-col gap-1">
            <Label value="Email" className="text-lg" />
            <TextInput
              required
              type="email"
              sizing="lg"
              placeholder="example@example.com"
              id="email"
              autoComplete="off"
              onChange={handleChange}
            />
          </div>
          <div className="flex flex-col gap-1">
            <Label value="Password" className="text-lg" />
            <TextInput
              required
              type="password"
              sizing="lg"
              placeholder="Password"
              id="password"
              onChange={handleChange}
            />
          </div>
          <Button
            variant="contained"
            style={{ backgroundColor: "#26597C", color: "#FFFFFF" }}
            type="submit"
            size="large"
            disabled={loading}
          >
            {loading ? (
              <>
                <Spinner size="sm" />
                <span className="pl-3">Loading...</span>
              </>
            ) : (
              "Sign In"
            )}
          </Button>
        </form>
        <div className="flex gap-2 text-sm mt-4">
          <span>Do not have an account?</span>
          <Link to={"/sign-up"} className="text-blue-500">
            Sign Up
          </Link>
        </div>
        {errorMessage && (
          <Alert severity="error" className="mt-4 border border-red-600">
            {errorMessage}
          </Alert>
        )}
      </div>
    </div>
  );
}
