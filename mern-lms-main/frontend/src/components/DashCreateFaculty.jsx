import { Alert, Button } from "@mui/material";
import { Label, TextInput } from "flowbite-react";
import { useState } from "react";

export default function DashCreateFaculty() {
  const [formData, setFormData] = useState({});
  const [createError, setCreateError] = useState(null);
  const [createSuccess, setCreateSuccess] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setCreateError(null);
    setCreateSuccess(null);
    try {
      const res = await fetch("/api/faculty/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      const data = await res.json();
      if (res.ok) {
        setCreateSuccess("Create faculty successfully!");
      } else {
        setCreateError(data.message);
      }
      setLoading(false);
    } catch (error) {
      setCreateError(error.message);
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-4 w-full">
      <h1 className="my-8 text-center font-bold text-4xl">Create Faculty</h1>
      <form className="flex flex-col gap-4" onSubmit={handleSubmit}>
        <div className="flex flex-col gap-2">
          <Label value="Faculty Name" className="text-lg" />
          <TextInput
            type="text"
            placeholder="Enter faculty name (e.g, Computer Science and Engineering)"
            required
            id="name"
            sizing="lg"
            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
          />
        </div>
        <Button
          variant="contained"
          style={{
            backgroundColor: "#26597C",
            color: "#FFFFFF",
            textTransform: "none",
          }}
          type="submit"
          size="large"
          disabled={loading}
        >
          Create Faculty
        </Button>
        {createError && (
          <Alert severity="error" className="border border-red-600">
            {createError}
          </Alert>
        )}
        {createSuccess && (
          <Alert severity="success" className="border border-green-600">
            {createSuccess}
          </Alert>
        )}
      </form>
    </div>
  );
}
