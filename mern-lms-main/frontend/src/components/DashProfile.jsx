import { useEffect, useRef, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  getDownloadURL,
  getStorage,
  ref,
  uploadBytesResumable,
} from "firebase/storage";
import { app } from "../firebase";
import { CircularProgressbar } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";
import {
  updateStart,
  updateSuccess,
  updateFailure,
} from "../redux/user/userSlice";
import { Alert, Button, Select, MenuItem, FormControl } from "@mui/material";
import { DatePicker } from "@mui/x-date-pickers/DatePicker";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";
import { LocalizationProvider } from "@mui/x-date-pickers/LocalizationProvider";
import dayjs from "dayjs";
import { Label, TextInput } from "flowbite-react";

export default function DashProfile() {
  const { currentUser } = useSelector((state) => state.user);
  const [faculties, setFaculties] = useState([]);
  const [imageFile, setImageFile] = useState(null);
  const [imageFileUrl, setImageFileUrl] = useState(null);
  const [imageFileUploadProgress, setImageFileUploadProgress] = useState(null);
  const [imageFileUploadError, setImageFileUploadError] = useState(null);
  const [imageFileUploading, setImageFileUploading] = useState(false);
  const [updateUserSuccess, setUpdateUserSuccess] = useState(null);
  const [updateUserError, setUpdateUserError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({ ...currentUser });
  const filePickerRef = useRef();
  const dispatch = useDispatch();

  useEffect(() => {
    const fetchFaculties = async () => {
      try {
        const res = await fetch(`/api/faculty/get`);
        const data = await res.json();
        if (res.ok) {
          setFaculties(data);
        }
      } catch (error) {
        console.log(error.message);
      }
    };
    fetchFaculties();
  }, []);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      setImageFileUrl(URL.createObjectURL(file));
    }
  };

  useEffect(() => {
    if (imageFile) {
      uploadImage();
    }
  }, [imageFile]);

  const uploadImage = async () => {
    setImageFileUploading(true);
    setImageFileUploadError(null);
    const storage = getStorage(app);
    const fileName = new Date().getTime() + imageFile.name;
    const storageRef = ref(storage, fileName);
    const uploadTask = uploadBytesResumable(storageRef, imageFile);
    uploadTask.on(
      "state_changed",
      (snapshot) => {
        const progress =
          (snapshot.bytesTransferred / snapshot.totalBytes) * 100;
        setImageFileUploadProgress(progress.toFixed(0));
      },
      () => {
        setImageFileUploadError("Could not upload image!");
        setImageFileUploadProgress(null);
        setImageFile(null);
        setImageFileUrl(null);
        setImageFileUploading(false);
      },
      () => {
        getDownloadURL(uploadTask.snapshot.ref).then((downloadUrl) => {
          setImageFileUrl(downloadUrl);
          setFormData({ ...formData, profilePicture: downloadUrl });
          setImageFileUploading(false);
        });
      }
    );
  };

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.id]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setUpdateUserSuccess(null);
    setUpdateUserError(null);
    if (Object.keys(formData).length === 0) {
      setUpdateUserError("No changes were made!");
      setLoading(false);
      return;
    }
    if (imageFileUploading) {
      setUpdateUserError("Please wait for image to upload!");
      setLoading(false);
      return;
    }
    try {
      dispatch(updateStart());
      const res = await fetch(`/api/user/update/${currentUser?._id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      const data = await res.json();
      if (!res.ok) {
        dispatch(updateFailure(data.message));
        setUpdateUserError(data.message);
      } else {
        dispatch(updateSuccess(data));
        setUpdateUserSuccess("User's profile updated successfully!");
      }
      setLoading(false);
    } catch (error) {
      dispatch(updateFailure(error.message));
      setUpdateUserError(error.message);
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-4 w-full">
      <h1 className="my-8 text-center font-bold text-4xl">Profile</h1>
      <input
        type="file"
        accept="image/*"
        onChange={handleImageChange}
        ref={filePickerRef}
        hidden
      />
      <div
        className="relative w-36 h-36 self-center cursor-pointer shadow-md overflow-hidden rounded-full mx-auto mb-6"
        onClick={() => filePickerRef.current.click()}
      >
        {imageFileUploadProgress && (
          <CircularProgressbar
            value={imageFileUploadProgress || 0}
            text={`${imageFileUploadProgress}%`}
            strokeWidth={5}
            styles={{
              root: {
                width: "100%",
                height: "100%",
                position: "absolute",
                top: 0,
                left: 0,
              },
              path: {
                stroke: `rgba(62, 152, 199, ${imageFileUploadProgress / 100})`,
              },
            }}
          />
        )}
        <img
          src={imageFileUrl || currentUser?.profilePicture}
          alt="User avatar"
          className={`rounded-full w-full h-full object-cover border-8 border-[lightgray] ${
            imageFileUploadProgress &&
            imageFileUploadProgress < 100 &&
            "opacity-60"
          }`}
        />
      </div>
      {imageFileUploadError && (
        <Alert severity="error" className="mb-4 border border-red-600">
          {imageFileUploadError}
        </Alert>
      )}
      <form onSubmit={handleSubmit} className="space-y-6">
        <LocalizationProvider dateAdapter={AdapterDayjs}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex flex-col gap-2">
              <Label value="Email" className="text-lg" />
              <TextInput
                required
                id="email"
                sizing="lg"
                value={formData?.email}
                onChange={handleChange}
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label value="Password" className="text-lg" />
              <TextInput
                id="password"
                type="password"
                sizing="lg"
                onChange={handleChange}
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label value="Name" className="text-lg" />
              <TextInput
                id="name"
                sizing="lg"
                value={formData?.name}
                onChange={handleChange}
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label value="Date of Birth" className="text-lg" />
              <DatePicker
                value={
                  formData?.dateOfBirth ? dayjs(formData?.dateOfBirth) : null
                }
                onChange={(newValue) =>
                  setFormData({
                    ...formData,
                    dateOfBirth: newValue?.format("YYYY-MM-DD"),
                  })
                }
                slotProps={{ textField: { fullWidth: true, size: "large" } }}
                format="DD/MM/YYYY"
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label value="Gender" className="text-lg" />
              <FormControl fullWidth>
                <Select
                  id="gender"
                  value={formData?.gender}
                  onChange={(e) =>
                    setFormData({ ...formData, gender: e.target.value })
                  }
                  size="large"
                >
                  <MenuItem value="male">Male</MenuItem>
                  <MenuItem value="female">Female</MenuItem>
                  <MenuItem value="other">Other</MenuItem>
                </Select>
              </FormControl>
            </div>
            <div className="flex flex-col gap-2">
              <Label value="Address" className="text-lg" />
              <TextInput
                id="address"
                sizing="lg"
                value={formData?.address}
                onChange={handleChange}
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label value="Phone Number" className="text-lg" />
              <TextInput
                id="phoneNumber"
                sizing="lg"
                value={formData?.phoneNumber}
                onChange={handleChange}
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label value="Enrolled Year" className="text-lg" />
              <FormControl fullWidth>
                <Select
                  id="enrolledYear"
                  value={formData?.enrolledYear}
                  onChange={(e) =>
                    setFormData({ ...formData, enrolledYear: e.target.value })
                  }
                  size="large"
                >
                  {[2020, 2021, 2022, 2023, 2024, 2025].map((year) => (
                    <MenuItem key={year} value={year}>
                      {year}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </div>
            <div className="flex flex-col gap-2">
              <Label value="Faculty" className="text-lg" />
              <FormControl fullWidth>
                <Select
                  value={formData?.facultyId}
                  onChange={(e) =>
                    setFormData({ ...formData, facultyId: e.target.value })
                  }
                >
                  {faculties?.length > 0 &&
                    faculties.map((faculty) => (
                      <MenuItem key={faculty?._id} value={faculty?._id}>
                        {faculty?.name}
                      </MenuItem>
                    ))}
                </Select>
              </FormControl>
            </div>
            {currentUser?.isAdmin && (
              <div className="flex flex-col gap-2">
                <Label value="Admin ID" className="text-lg" />
                <TextInput
                  id="adminId"
                  sizing="lg"
                  value={formData?.adminId}
                  onChange={handleChange}
                />
              </div>
            )}
            {currentUser?.isEducator && (
              <div className="flex flex-col gap-2">
                <Label value="Educator ID" className="text-lg" />
                <TextInput
                  id="educatorId"
                  sizing="lg"
                  value={formData?.educatorId}
                  onChange={handleChange}
                />
              </div>
            )}
            {currentUser?.isStudent && (
              <div className="flex flex-col gap-2">
                <Label value="Student ID" className="text-lg" />
                <TextInput
                  id="studentId"
                  sizing="lg"
                  value={formData?.studentId}
                  onChange={handleChange}
                />
              </div>
            )}
          </div>
        </LocalizationProvider>
        <Button
          variant="contained"
          style={{
            backgroundColor: "#26597C",
            color: "#fff",
            textTransform: "none",
          }}
          type="submit"
          size="large"
          disabled={loading || imageFileUploading}
        >
          {loading ? "Updating..." : "Update Profile"}
        </Button>
        {updateUserSuccess && (
          <Alert severity="success" className="mt-4 border border-green-600">
            {updateUserSuccess}
          </Alert>
        )}
        {updateUserError && (
          <Alert severity="error" className="mt-4 border border-red-600">
            {updateUserError}
          </Alert>
        )}
      </form>
    </div>
  );
}
